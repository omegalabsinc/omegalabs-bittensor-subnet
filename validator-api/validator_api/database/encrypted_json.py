from cryptography.fernet import Fernet, InvalidToken
import json
from typing import Optional, Union

from sqlalchemy.types import TypeDecorator, LargeBinary
from sqlalchemy.engine.interfaces import Dialect
from pydantic import BaseModel

from validator_api.config import ENCRYPTION_KEY


fernet = Fernet(ENCRYPTION_KEY)

# Type alias for any valid JSON type, including Pydantic BaseModel
JSONType = Union[dict, list, str, int, float, bool, None, BaseModel]


class EncryptedJSON(TypeDecorator):
    # For MySQL, the default limit here is 64 kb. In the prod DB, I (Salman) set it to 4GB.
    impl = LargeBinary

    def process_bind_param(
        self, value: Optional[JSONType], dialect: Dialect
    ) -> Optional[bytes]:
        if value is not None:
            try:
                return encrypt_data(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Error encrypting data: {str(e)}")
        return None

    def process_result_value(
        self, value: Optional[bytes], dialect: Dialect
    ) -> Optional[JSONType]:
        if value is not None:
            try:
                return decrypt_data(value)
            except (InvalidToken, json.JSONDecodeError) as e:
                raise ValueError(f"Error decrypting data: {str(e)}")
        return None


def encrypt_data(data: JSONType) -> bytes:
    try:
        if isinstance(data, BaseModel):
            data = json.loads(data.model_dump_json())
        return fernet.encrypt(json.dumps(data).encode())
    except (TypeError, ValueError) as e:
        raise ValueError(f"Error encoding or encrypting data: {str(e)}")


def decrypt_data(encrypted_data: bytes) -> JSONType:
    try:
        decrypted_data = fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
    except InvalidToken:
        raise ValueError("Invalid token or key used for decryption")
    except json.JSONDecodeError:
        raise ValueError("Decrypted data is not valid JSON")


class LargeEncryptedJSON(EncryptedJSON):
    impl = LargeBinary(
        length=4 * 1024 * 1024 * 1024 - 1
    )  # 4 GB - 1 byte because thats the MySQL max


class MediumEncryptedJSON(EncryptedJSON):
    impl = LargeBinary(
        length=16 * 1024 * 1024 - 1
    )  # 16 MB - 1 byte (MySQL MEDIUMBLOB max size)


def test_encrypted_json():
    encrypted_json_type = EncryptedJSON()

    class FakeModel(BaseModel):
        name: str
        value: int

    class NestedFakeModel(BaseModel):
        nested: FakeModel

    # Test with different JSON types
    test_cases = [
        {"key": "value"},  # dict
        ["item1", "item2"],  # list
        "string",  # str
        42,  # int
        3.14,  # float
        True,  # bool
        None,  # null
        {
            "nested": {"list": [1, 2, 3], "dict": {"a": 1, "b": 2}}
        },  # complex nested structure
        FakeModel(name="Test", value=123),  # Pydantic BaseModel
        NestedFakeModel(
            nested=FakeModel(name="Nested", value=456)
        ),  # Nested Pydantic BaseModel
    ]

    for case in test_cases:
        # Simulate database write
        encrypted = encrypted_json_type.process_bind_param(case, None)

        # Simulate database read
        decrypted = encrypted_json_type.process_result_value(encrypted, None)

        if isinstance(case, BaseModel):
            assert type(case)(**decrypted) == case, f"Failed for case: {case}"
        else:
            assert decrypted == case, f"Failed for case: {case}"
        print(f"Success: {case}")


if __name__ == "__main__":
    test_encrypted_json()
