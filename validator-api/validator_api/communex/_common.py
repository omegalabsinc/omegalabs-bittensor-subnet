import random

class ComxSettings():
    # TODO: improve node lists
    NODE_URLS: list[str] = [
        "wss://commune.api.onfinality.io/public-ws",
        "wss://commune.api.onfinality.io/public-ws",
        "wss://commune.api.onfinality.io/public-ws",
        "wss://commune.api.onfinality.io/public-ws",
        "wss://commune.api.onfinality.io/public-ws",
        "wss://commune-api-node-0.communeai.net",
        "wss://commune-api-node-1.communeai.net",
        "wss://commune-api-node-2.communeai.net",
        "wss://commune-api-node-3.communeai.net",
        "wss://commune-api-node-4.communeai.net",
        "wss://commune-api-node-5.communeai.net",
        "wss://commune-api-node-6.communeai.net",
        "wss://commune-api-node-7.communeai.net",
        "wss://commune-api-node-8.communeai.net",
        "wss://commune-api-node-9.communeai.net",
        "wss://commune-api-node-10.communeai.net",
        "wss://commune-api-node-11.communeai.net",
        "wss://commune-api-node-12.communeai.net",
        "wss://commune-api-node-13.communeai.net",
        "wss://commune-api-node-14.communeai.net",
        "wss://commune-api-node-15.communeai.net",
        "wss://commune-api-node-16.communeai.net",
        "wss://commune-api-node-17.communeai.net",
        "wss://commune-api-node-18.communeai.net",
        "wss://commune-api-node-19.communeai.net",
        "wss://commune-api-node-20.communeai.net",
        "wss://commune-api-node-21.communeai.net",
        "wss://commune-api-node-22.communeai.net",
        "wss://commune-api-node-23.communeai.net",
        "wss://commune-api-node-24.communeai.net",
        "wss://commune-api-node-25.communeai.net",
        "wss://commune-api-node-26.communeai.net",
        "wss://commune-api-node-27.communeai.net",
        "wss://commune-api-node-28.communeai.net",
        "wss://commune-api-node-29.communeai.net",
        "wss://commune-api-node-30.communeai.net",
        "wss://commune-api-node-31.communeai.net",
    ]
    TESTNET_NODE_URLS: list[str] = [
        "wss://testnet-commune-api-node-0.communeai.net"]


def get_node_url(
    comx_settings: ComxSettings | None = None, *, use_testnet: bool = False
) -> str:
    comx_settings = comx_settings or ComxSettings()
    match use_testnet:
        case True:
            node_url = random.choice(comx_settings.TESTNET_NODE_URLS)
        case False:
            node_url = random.choice(comx_settings.NODE_URLS)
    return node_url