import secrets

def generate_api_key():
    return secrets.token_urlsafe(32)  # Generates a 32-byte (256-bit) key

new_api_key = generate_api_key()
print(new_api_key)