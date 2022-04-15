import preprocessor
import os
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoModelWithLMHead,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class Model:
    def __init__(self):
        """
        Constructor for model. Uses CPU by default for reproducibility, though I actually trained this on g-collab
        using a single GPU.
        """
        self.tokenizer = AutoTokenizer.from_pretrained('DialoGPT-small')
        self.device = torch.device('cpu')
        # Load the pretrained DialoGPT transformer model.
        self.model = AutoModelWithLMHead.from_pretrained('DialoGPT-small', from_tf=False)
        # And send it to the (C/G)PU.
        self.model.to(self.device)

    def train(self, dataset: preprocessor.ConversationData) -> (int, float):
        """
        Given a preprocessed conversation dataset consisting of tokenized conversations, retrain the model to predict
        these particular conversations.
        :param dataset: A set of tokenized sentences from conversations, delimited by EOS tokens.
        :return: (Num steps, avg. loss per step)
        """
        num_epochs = 3
        sampler = RandomSampler(dataset)

        def collate(tokens: [torch.Tensor]):
            """
            Append 0s to tensor so as to increase length of token vector to that required by model.
            :param tokens: Non-padded token vector.
            :return: Padded token vector.
            """
            return torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=4, collate_fn=collate, drop_last=True)
        print(len(dataset))
        num_rows = len(dataset) * 3
        self.model.resize_token_embeddings(len(self.tokenizer))

        # As stated previously, we're retraining an already-trained model, so we can't just declare our own optimization
        # parameters. We need to use reuse the parameters of the model (though we can drop layer normalization, since
        # that has a bad performance impact with sufficiently large datasets).
        optimizer = AdamW([{
            "params": [param for name, param in self.model.named_parameters()
                       if not ("bias" in name or "LayerNorm.weight" in name)],
            "weight_decay": 0
        }, {
            "params": [param for name, param in self.model.named_parameters()
                       if ("bias" in name or "LayerNorm.weight" in name)],
            "weight_decay": 0
        }])

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_rows)

        global_step = current_epoch = 0
        total_loss = 0.0
        # We need to set NN weights to 0 before every backprop call.
        self.model.zero_grad()

        # This has the nice effect of printing progress bar.
        for _ in trange(current_epoch, num_epochs, desc="Epoch"):
            for step, batch in enumerate(tqdm(dataloader, desc="Step")):
                # Model accepts vectors of exactly 1024 tokens. If this vector exceeds that, skip it.
                if batch.shape[1] > 1024:
                    continue

                # Calculate loss from given batch, backpropagate loss.
                inputs = batch.to(self.device)
                labels = batch.to(self.device)
                self.model.train()
                # Most pytorch models
                out = self.model(inputs, labels=labels)
                loss = out[0]
                loss.backward()
                total_loss += loss.item()

                # Perform gradient normalization. This is essentially a mechanism to prevent exploding gradients.
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                # Perform a single SGD step.
                optimizer.step()
                scheduler.step()
                # Set gradients to 0 for next loop (before backprop).
                self.model.zero_grad()
                global_step += 1

                # Periodically save model at checkpoint, in case some error happens or process gets killed.
                if global_step % 2000 == 0:
                    print('Saving model')
                    output = f'./checkpt-{global_step}'
                    self.save_model(output)
                    torch.save(optimizer.state_dict(), f'{output}/optimizer.pt')
                    torch.save(scheduler.state_dict(), f'{output}/scheduler.pt')

        self.save_model(f'./output-medium')
        return global_step, total_loss / global_step

    def save_model(self, output_dir: str):
        """
        Save the model at its current stage in training to the specified directory.
        :param output_dir: Directory to which model will be saved.
        """
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def evaluate(self, dataset: preprocessor.ConversationData):
        """
        Given a test dataset, compute the perplexity of the model.
        :param dataset: A dataset consisting of tokenized conversations.
        :return: Perplexity of model. Essentially a metric of how confident the model is in the
                 probability of responses in the test dataset. Ideally, it should be very confident.
                 Lower scores are better.
        """
        def collate(tokens: [torch.Tensor]):
            """
            Standard tensor-padding func.
            """
            return torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)

        # Get sampler/loader to randomly select data from dataset.
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=4, collate_fn=collate, drop_last=True)

        loss = 0.0
        eval_steps = 0
        self.model.eval()

        # Calculate cumulative loss.
        for batch in tqdm(dataloader, desc="Evaluation steps."):
            inputs = batch.to(self.device)
            labels = batch.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs, labels=labels)
                batch_loss = outputs[0]
                if __name__ == '__main__':
                    loss += batch_loss.mean().item()

            eval_steps += 1

        loss = loss / eval_steps
        # Now do nth-root(1/[AVG_LOSS])
        return torch.exp(torch.tensor(loss))


if __name__ == '__main__':
    """"
    Train model, compute perplexity.
    This assumes you've already run preprocessor.py and generated train/test JSON files.
    If not, you'll get an error, obviously.
    """
    model = Model()
    train = preprocessor.ConversationData(json_path="./training/training.json", cutoff=10000)
    model.train(train)
    test = preprocessor.ConversationData(json_path="./training/testing.json", cutoff=10000)
    perplexity = model.evaluate(test)
    print("Perplexity is %d", perplexity)

