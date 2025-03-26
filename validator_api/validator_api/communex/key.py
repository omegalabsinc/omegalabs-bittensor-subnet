from typing import TypeGuard

from substrateinterface import Keypair  # type: ignore
from substrateinterface.utils import ss58  # type: ignore

from validator_api.validator_api.communex.types import Ss58Address


def is_ss58_address(address: str, ss58_format: int = 42) -> TypeGuard[Ss58Address]:
    """
    Validates whether the given string is a valid SS58 address.

    Args:
        address: The string to validate.
        ss58_format: The SS58 format code to validate against.

    Returns:
        True if the address is valid, False otherwise.
    """

    return ss58.is_valid_ss58_address(address, valid_ss58_format=ss58_format)


def check_ss58_address(
    address: str | Ss58Address, ss58_format: int = 42
) -> Ss58Address:
    """
    Validates whether the given string is a valid SS58 address.

    Args:
        address: The string to validate.
        ss58_format: The SS58 format code to validate against.

    Returns:
        The validated SS58 address.

    Raises:
        AssertionError: If the address is invalid.
    """

    assert is_ss58_address(address, ss58_format), f"Invalid SS58 address '{address}'"
    return Ss58Address(address)


def generate_keypair() -> Keypair:
    """
    Generates a new keypair.
    """
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    return keypair
