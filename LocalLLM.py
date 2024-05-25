from transformers import GPT2Tokenizer, GPT2LMHeadModel

def initialize_model(model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def truncate_context(context, tokenizer, max_length):
    context_tokens = tokenizer.encode(context)
    if len(context_tokens) > max_length:
        context_tokens = context_tokens[-max_length:]
    return tokenizer.decode(context_tokens, clean_up_tokenization_spaces=True)

def generate_response(tokenizer, model, context, question, max_new_tokens=50):
    context = truncate_context(context, tokenizer, max_length=1024 - max_new_tokens - len(tokenizer.encode(question)))
    
    input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response.strip()