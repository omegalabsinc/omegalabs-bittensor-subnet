"""
Common types for the communex module.
"""

from typing import NewType, TypedDict

Ss58Address = NewType("Ss58Address", str)
"""Substrate SS58 address.

The `SS58 encoded address format`_ is based on the Bitcoin Base58Check format,
but with a few modification specifically designed to suite Substrate-based
chains.

.. _SS58 encoded address format:
    https://docs.substrate.io/reference/address-formats/
"""


# TODO: replace with dataclasses


class NetworkParams(TypedDict):
    max_allowed_modules: int
    max_registrations_per_block: int
    target_registrations_interval: int  #  in blocks
    target_registrations_per_interval: int
    unit_emission: int
    max_name_length: int
    min_name_length: int
    burn_rate: int
    min_burn: int  # min burn to register
    max_burn: int  # max burn to register
    min_stake: int
    min_weight_stake: int
    max_allowed_subnets: int
    adjustment_alpha: int
    floor_delegation_fee: int
    max_allowed_weights: int
    curator: Ss58Address
    proposal_cost: int
    proposal_expiration: int
    proposal_participation_threshold: int
    subnet_stake_threshold: int


class SubnetParams(TypedDict):
    founder: Ss58Address
    founder_share: int
    immunity_period: int
    incentive_ratio: int
    max_allowed_uids: int
    max_allowed_weights: int
    min_allowed_weights: int
    max_stake: int
    max_weight_age: int
    min_stake: int
    name: str
    tempo: int
    trust_ratio: int
    vote_mode: str
    bonds_ma: int | None
    maximum_set_weight_calls_per_epoch: int | None


# redundant "TypedDict" inheritance because of pdoc warns.
# see https://github.com/mitmproxy/pdoc/blob/26d40827ddbe1658e8ac46cd092f17a44cf0287b/pdoc/doc.py#L691-L692
class SubnetParamsWithEmission(SubnetParams, TypedDict):
    """SubnetParams with emission field.
    """
    emission: int
    """Subnet emission percentage (0-100).
    """


class ModuleInfo(TypedDict):
    uid: int
    key: Ss58Address
    name: str
    address: str  # "<ip>:<port>"
    emission: int
    incentive: int
    dividends: int
    stake_from: list[tuple[str, int]]  # TODO: type key with Ss58Address
    regblock: int  # block number
    last_update: int  # block number
    stake: int
    delegation_fee: int
    metadata: str


class ModuleInfoWithBalance(ModuleInfo):
    balance: int


class ModuleInfoWithOptionalBalance(ModuleInfo):
    balance: int | None