from typing import Dict

import torch


class PolicyGradientLoss:
    """Policy gradient loss for both off and on-policy training.

    Input Arguments:
        model_output_key: logits name in the model outputs.
        labels_key: labels name in the batch used for computing the policy probability.
        sampling_log_prob_key: key for sample negative log probabilities from the sampling policy.
        reward_key: reward key in batch.
        use_importance_sampling: if importance sampling is used for loss (e.g. is this off or on policy).
    """

    def __init__(
        self,
        model_output_key: str = "logits",
        labels_key: str = "labels",
        sampling_log_prob_key: str = "log_probs",
        reward_key: str = "reward",
        use_importance_sampling: bool = False,
    ):
        self.model_output_key = model_output_key
        self.labels_key = labels_key
        self.sampling_log_prob_key = sampling_log_prob_key
        self.reward_key = reward_key
        self.use_importance_sampling = use_importance_sampling
        self.cross_entropy = torch.nn.CrossEntropyLoss(
            ignore_index=IGNORE_TOKEN_ID, reduction="none"
        )

    def __call__(
        self, predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:

        logits = predictions[self.model_output_key]
        labels = batch[self.labels_key]

        shifted_logits = logits[:, :-1, ...].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        rewards = batch[self.reward_key]

        log_policy = -(
            self.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1),
            )
            .view(*shifted_labels.shape)
            .sum(dim=1)
        )

        if self.use_importance_sampling:
            # Detach importance sampling weights as the policy gradient is "grad J = weight * grad log (P) * R".
            # importance_weights = torch.exp(log_policy) / torch.exp(-batch[self.sampling_log_prob_key]).detach()
            old_log_policy = batch[self.sampling_log_prob_key]
            importance_weights = torch.exp(log_policy - old_log_policy).detach()
            return -(importance_weights * log_policy * rewards).mean()
        else:
            return -(log_policy * rewards).mean()


def train_step_fn(self):
    def fun(backend, batch, loss_fn):

        backend.init_step()
        # input_ids: generated ids, the id that get selected (action that model actually takes)
        input_ids = backend.to_backend_device(batch["input_ids"])

        # logit: raw output of the model
        model_output = backend(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )

        loss = loss_fn(model_output, batch)
        backend.backward(loss)
        backend.step()
        print("LOSS_IN_STEP:", loss)
        return loss

    return fun


class BatchBuilder(ABC):
    """Abstract class for sampling policy and evaluating rewards."""

    def __init__(self, backend: Backend, environment: RLEnvironment):
        self.backend = backend
        self.environment = environment

    def initialize(self):
        self.environment.initialize()

    @abstractmethod
    def build(
        self, batch: Dict[str, Any], num_of_retrieved_doc: int
    ) -> Iterable[Dict[str, Any]]:
        return [batch]


class SampleBatchBuilder(BatchBuilder):
    """Generating samples for a batch"""

    def __init__(
        self,
        tokenizer: Any,  # for evaluating the LLM output
        backend: Backend,
        environment: RLEnvironment,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        max_new_tokens: int,
        sample_size: int,
        train_batch_size: int,
        decoding_batch_size: int = 1,
        epochs: int = 1,
        temperature: float = 1.0,
        input_arguments_override: Tuple[str, ...] = (
            "input_ids_for_generation",
            "attention_mask_for_generation",
        ),
        keep_fields: Tuple[str, ...] = tuple(),
        synced_gpus: bool = False,
    ):
        super().__init__(backend, environment)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_new_tokens = max_new_tokens
        self.sample_size = sample_size
        self.train_batch_size = train_batch_size
        self.decoding_batch_size = decoding_batch_size
        self.epochs = epochs
        self.temperature = temperature
        self.model_input_arg_names = input_arguments_override
        self.keep_fields = keep_fields
        self.synced_gpus = synced_gpus
        self.backend = backend
        self.environment = environment
        self.tokenizer = tokenizer
        self.reward_baseline = 0.5

    def build(
        self, batch: Dict[str, Any], num_of_retrieved_doc: int = 10000
    ) -> Iterable[Dict[str, Any]]:
        samples: List[Dict[str, torch.Tensor]] = []

        # Data loader yielding batched for decoding
        decoding_data_loader = torch.utils.data.DataLoader(
            dataset=convert_dict_to_list(batch),
            batch_size=self.decoding_batch_size,
            num_workers=0,
        )

        # Decode possible continuation to prompts in batch using the current policy, and save the samples.
        loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.pad_token_id, reduction="none"
        )
        with model_mode_context(
            model=self.backend, target_mode=ModelMode.EVALUATION
        ), torch.no_grad():
            for decoding_batch in decoding_data_loader:

                inputs_for_generation = {
                    key.replace("_for_generation", ""): decoding_batch[key]
                    for key in self.model_input_arg_names
                }

                inputs_for_generation["input_ids"] = inputs_for_generation[
                    "input_ids"
                ].to(self.backend.device)

                # model will generate self.sample_size of candidates, shape: [sample size(num of candidate), generated index]
                model_outputs = self.backend.unwrapped_model.generate(
                    **inputs_for_generation,
                    num_return_sequences=self.sample_size,
                    temperature=self.temperature,
                    bos_token_id=self.bos_token_id,
                    eos_token_id=self.eos_token_id,
                    synced_gpus=self.synced_gpus,
                    do_sample=True,
                    num_beams=1,
                    max_new_tokens=self.max_new_tokens,
                )

                input_size = decoding_batch["attention_mask_for_generation"].shape[1]
                generated_output = model_outputs[:, input_size:].contiguous()
                generated_mask = (generated_output != self.pad_token_id).to(
                    dtype=decoding_batch["attention_mask_for_generation"].dtype,
                    device=decoding_batch["attention_mask_for_generation"].device,
                )

                inputs_mask = decoding_batch[
                    "attention_mask_for_generation"
                ].repeat_interleave(repeats=self.sample_size, dim=0)
                attention_mask = torch.cat(
                    [
                        inputs_mask,
                        generated_mask,
                    ],
                    dim=1,
                )

                # HF models don't expose the real log-probs so logits must be evaluated separately.
                logits = self.backend(
                    input_ids=model_outputs,
                    attention_mask=attention_mask,
                    return_dict=True,
                )["logits"]

                generated_logits = logits[:, (input_size - 1) : -1, ...].contiguous()
                scores = -(
                    loss_fn(
                        generated_logits.view(-1, generated_logits.shape[-1]),
                        generated_output.view(-1),
                    )
                    .view(*generated_output.shape)
                    .sum(dim=1)
                )

                generated_labels = generated_output.clone()
                generated_labels[generated_labels == self.pad_token_id] = (
                    IGNORE_TOKEN_ID
                )
                labels = torch.cat(
                    [
                        torch.ones_like(inputs_mask, device=self.backend.device)
                        * IGNORE_TOKEN_ID,
                        generated_labels,
                    ],
                    dim=1,
                )

                targets = decoding_batch[
                    self.environment.TARGET_NAME
                ].repeat_interleave(repeats=self.sample_size, dim=0)

                rewards = self.environment.reward(
                    {
                        self.environment.INPUT_NAME: generated_output,
                        self.environment.TARGET_NAME: targets,
                        **{
                            key: batch[key].repeat_interleave(
                                repeats=self.sample_size, dim=0
                            )
                            for key in self.environment.additional_input_names()
                        },
                    },
                    num_of_retrieved_doc,
                    mode="rank",
                )

                rewards["reward"] = rewards["reward"].to(self.backend.device)
                rewards["reward"] = rewards["reward"] - self.reward_baseline
                model_outputs = model_outputs.to(self.backend.device)

                samples.extend(
                    convert_dict_to_list(
                        {
                            "log_probs": scores,  # i.e., old_logprob
                            "input_ids": model_outputs,
                            "attention_mask": attention_mask,
                            "labels": labels,
                            **rewards,
                            **{
                                key: batch[key].repeat_interleave(
                                    repeats=self.sample_size, dim=0
                                )
                                for key in self.keep_fields
                            },
                        }
                    )
                )
        # Yield training batches for the model.
        yield from torch.utils.data.DataLoader(
            dataset=samples,
            collate_fn=NbestCollator(
                padding_lookup={
                    "input_ids": torch.tensor(
                        self.pad_token_id,
                        dtype=model_outputs.dtype,
                        device=model_outputs.device,
                    ),
                    "labels": torch.tensor(
                        IGNORE_TOKEN_ID, dtype=labels.dtype, device=labels.device
                    ),
                    "attention_mask": torch.tensor(
                        0,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    self.environment.INPUT_NAME: torch.tensor(
                        self.pad_token_id,
                        dtype=rewards[self.environment.INPUT_NAME].dtype,
                        device=rewards[self.environment.INPUT_NAME].device,
                    ),
                },
            ),
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0,
        )
