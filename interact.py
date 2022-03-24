from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("output-medium")
    model = AutoModelWithLMHead.from_pretrained("output-medium")
    for i in range(5):
        new_user_input_ids = tokenizer.encode(input(">> User: ") + tokenizer.eos_token, return_tensors='pt')
        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if i > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens
        # will concatenate to this.
        chat_history_ids = model.generate(
            bot_input_ids, max_length=10000,
            pad_token_id=0
        )
        # for id in chat_history_ids[-1]:
        #     if id == 0:
        #         break
        #     print(tokenizer.decode(id), end=' ')
        # print(chat_history_ids)
        # # pretty print last ouput tokens from bot
        print("DialoGPT: {}".format(
            tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
