def text_validation(text):
    if not isinstance(text,str):
        raise TypeError('Input must be text!')

    if len(text) > 5000:
        raise ValueError('Input Exceed 5000 characters!')
    

    clean_text = text.strip()

    if not clean_text:
        raise ValueError('Input should no be empty!')

    
    return clean_text




text = text_validation(raw_text)
print(text)