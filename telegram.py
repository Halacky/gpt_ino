import telebot
bot = telebot.TeleBot('-')
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import GPT2Tokenizer
import torch

def create_model():
    # # Load a trained model and vocabulary that you have fine-tuned
    model = GPT2LMHeadModel.from_pretrained("C:\\Users\\RomeF\\yolov5\\res2")
    tokenizer = GPT2Tokenizer.from_pretrained("C:\\Users\\RomeF\\yolov5\\res2")
    model.to("cuda")
    model.eval()

    return model, tokenizer

def create_joke(message):
    # Generate Text

    model,tokenizer = create_model()
    prompt = message

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to("cuda")

    sample_outputs = model.generate(
                                    generated, 
                                    #bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=5, 
                                    max_length = 300,
                                    top_p=0.95, 
                                    num_return_sequences=1
                                    )

    return tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    joke = create_joke(message.text)
    
    bot.send_photo(message.from_user.id, photo=open('C:\\Users\\RomeF\\yolov5\\l-etKudYN1c.jpg', 'rb'))
    bot.send_message(message.from_user.id, joke)

bot.polling(none_stop=True, interval=0)