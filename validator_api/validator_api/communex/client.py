import json
import queue
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

from substrateinterface import ExtrinsicReceipt  # type: ignore
from substrateinterface import Keypair  # type: ignore
from substrateinterface import SubstrateInterface  # type: ignore
from substrateinterface.storage import StorageKey  # type: ignore

from validator_api.validator_api.communex.errors import (
    ChainTransactionError,
    NetworkQueryError,
)
from validator_api.validator_api.communex.types import (
    NetworkParams,
    Ss58Address,
    SubnetParams,
)

# TODO: InsufficientBalanceError, MismatchedLengthError etc

MAX_REQUEST_SIZE = 9_000_000


@dataclass
class Chunk:
    batch_requests: list[tuple[Any, Any]]
    prefix_list: list[list[str]]
    fun_params: list[tuple[Any, Any, Any, Any, str]]


T1 = TypeVar("T1")
T2 = TypeVar("T2")


class CommuneClient:
    """
    A client for interacting with Commune network nodes, querying storage,
    submitting transactions, etc.

    Attributes:
        wait_for_finalization: Whether to wait for transaction finalization.

    Example:
    ```py
    client = CommuneClient()
    client.query(name='function_name', params=['param1', 'param2'])
    ```

    Raises:
        AssertionError: If the maximum connections value is less than or equal
          to zero.
    """

    wait_for_finalization: bool
    _num_connections: int
    _connection_queue: queue.Queue[SubstrateInterface]

    def __init__(
        self,
        url: str,
        num_connections: int = 1,
        wait_for_finalization: bool = False,
    ):
        """
        Args:
            url: The URL of the network node to connect to.
            num_connections: The number of websocket connections to be opened.
        """
        assert num_connections > 0
        self._num_connections = num_connections
        self.wait_for_finalization = wait_for_finalization
        self._connection_queue = queue.Queue(num_connections)

        for _ in range(num_connections):
            self._connection_queue.put(SubstrateInterface(url))

    @property
    def connections(self) -> int:
        """
        Gets the maximum allowed number of simultaneous connections to the
        network node.
        """
        return self._num_connections

    @contextmanager
    def get_conn(self, timeout: float | None = None, init: bool = False):
        """
        Context manager to get a connection from the pool.

        Tries to get a connection from the pool queue. If the queue is empty,
        it blocks for `timeout` seconds until a connection is available. If
        `timeout` is None, it blocks indefinitely.

        Args:
            timeout: The maximum time in seconds to wait for a connection.

        Yields:
            The connection object from the pool.

        Raises:
            QueueEmptyError: If no connection is available within the timeout
              period.
        """
        conn = self._connection_queue.get(timeout=timeout)
        if init:
            conn.init_runtime()  # type: ignore
        try:
            yield conn
        finally:
            self._connection_queue.put(conn)

    def _get_storage_keys(
        self,
        storage: str,
        queries: list[tuple[str, list[Any]]],
        block_hash: str | None,
    ):
        send: list[tuple[str, list[Any]]] = []
        prefix_list: list[Any] = []

        key_idx = 0
        with self.get_conn(init=True) as substrate:
            for function, params in queries:
                storage_key = StorageKey.create_from_storage_function(  # type: ignore
                    storage,
                    function,
                    params,
                    runtime_config=substrate.runtime_config,
                    metadata=substrate.metadata,  # type: ignore
                )

                prefix = storage_key.to_hex()
                prefix_list.append(prefix)
                send.append(("state_getKeys", [prefix, block_hash]))
                key_idx += 1
        return send, prefix_list

    def _get_lists(
        self,
        storage_module: str,
        queries: list[tuple[str, list[Any]]],
        substrate: SubstrateInterface,
    ) -> list[tuple[Any, Any, Any, Any, str]]:
        """
        Generates a list of tuples containing parameters for each storage function based on the given functions and substrate interface.

        Args:
            functions (dict[str, list[query_call]]): A dictionary where keys are storage module names and values are lists of tuples.
                Each tuple consists of a storage function name and its parameters.
            substrate: An instance of the SubstrateInterface class used to interact with the substrate.

        Returns:
            A list of tuples in the format `(value_type, param_types, key_hashers, params, storage_function)` for each storage function in the given functions.

        Example:
            >>> _get_lists(
                    functions={'storage_module': [('storage_function', ['param1', 'param2'])]},
                    substrate=substrate_instance
                )
            [('value_type', 'param_types', 'key_hashers', ['param1', 'param2'], 'storage_function'), ...]
        """

        function_parameters: list[tuple[Any, Any, Any, Any, str]] = []
        metadata_pallet = substrate.metadata.get_metadata_pallet(  #  type: ignore
            storage_module
        )  # type: ignore
        for storage_function, params in queries:
            storage_item = metadata_pallet.get_storage_function(  #  type: ignore
                storage_function
            )  # type: ignore
            value_type = storage_item.get_value_type_string()  # type: ignore
            param_types = storage_item.get_params_type_string()  # type: ignore
            key_hashers = storage_item.get_param_hashers()  # type: ignore
            function_parameters.append(
                (value_type, param_types, key_hashers, params, storage_function)  # type: ignore
            )
        return function_parameters

    def _send_batch(
        self,
        batch_payload: list[Any],
        request_ids: list[int],
        extract_result: bool = True,
    ):
        """
        Sends a batch of requests to the substrate and collects the results.

        Args:
            substrate: An instance of the substrate interface.
            batch_payload: The payload of the batch request.
            request_ids: A list of request IDs for tracking responses.
            results: A list to store the results of the requests.
            extract_result: Whether to extract the result from the response.

        Raises:
            NetworkQueryError: If there is an `error` in the response message.

        Note:
            No explicit return value as results are appended to the provided 'results' list.
        """
        results: list[str | dict[Any, Any]] = []
        with self.get_conn(init=True) as substrate:
            try:
                substrate.websocket.send(  #  type: ignore
                    json.dumps(batch_payload)
                )  # type: ignore
            except NetworkQueryError:
                pass
            while len(results) < len(request_ids):
                received_messages = json.loads(substrate.websocket.recv())  # type: ignore
                if isinstance(received_messages, dict):
                    received_messages: list[dict[Any, Any]] = [received_messages]

                for message in received_messages:
                    if message.get("id") in request_ids:
                        if extract_result:
                            try:
                                results.append(message["result"])
                            except Exception:
                                raise (
                                    RuntimeError(
                                        f"Error extracting result from message: {message}"
                                    )
                                )
                        else:
                            results.append(message)
                    if "error" in message:
                        raise NetworkQueryError(message["error"])

            return results

    def _make_request_smaller(
        self,
        batch_request: list[tuple[T1, T2]],
        prefix_list: list[list[str]],
        fun_params: list[tuple[Any, Any, Any, Any, str]],
    ) -> tuple[list[list[tuple[T1, T2]]], list[Chunk]]:
        """
        Splits a batch of requests into smaller batches, each not exceeding the specified maximum size.

        Args:
            batch_request: A list of requests to be sent in a batch.
            max_size: Maximum size of each batch in bytes.

        Returns:
            A list of smaller request batches.

        Example:
            >>> _make_request_smaller(batch_request=[('method1', 'params1'), ('method2', 'params2')], max_size=1000)
            [[('method1', 'params1')], [('method2', 'params2')]]
        """
        assert len(prefix_list) == len(fun_params) == len(batch_request)

        def estimate_size(request: tuple[T1, T2]):
            """Convert the batch request to a string and measure its length"""
            return len(json.dumps(request))

        # Initialize variables
        result: list[list[tuple[T1, T2]]] = []
        current_batch = []
        current_prefix_batch = []
        current_params_batch = []
        current_size = 0

        chunk_list: list[Chunk] = []

        # Iterate through each request in the batch
        for request, prefix, params in zip(batch_request, prefix_list, fun_params):
            request_size = estimate_size(request)

            # Check if adding this request exceeds the max size
            if current_size + request_size > MAX_REQUEST_SIZE:
                # If so, start a new batch

                # Essentiatly checks that it's not the first iteration
                if current_batch:
                    chunk = Chunk(
                        current_batch, current_prefix_batch, current_params_batch
                    )
                    chunk_list.append(chunk)
                    result.append(current_batch)

                current_batch = [request]
                current_prefix_batch = [prefix]
                current_params_batch = [params]
                current_size = request_size
            else:
                # Otherwise, add to the current batch
                current_batch.append(request)
                current_size += request_size
                current_prefix_batch.append(prefix)
                current_params_batch.append(params)

        # Add the last batch if it's not empty
        if current_batch:
            result.append(current_batch)
            chunk = Chunk(current_batch, current_prefix_batch, current_params_batch)
            chunk_list.append(chunk)

        return result, chunk_list

    def _are_changes_equal(self, change_a: Any, change_b: Any):
        for (a, b), (c, d) in zip(change_a, change_b):
            if a != c or b != d:
                return False

    def _rpc_request_batch(
        self, batch_requests: list[tuple[str, list[Any]]], extract_result: bool = True
    ) -> list[str]:
        """
        Sends batch requests to the substrate node using multiple threads and collects the results.

        Args:
            substrate: An instance of the substrate interface.
            batch_requests : A list of requests to be sent in batches.
            max_size: Maximum size of each batch in bytes.
            extract_result: Whether to extract the result from the response message.

        Returns:
            A list of results from the batch requests.

        Example:
            >>> _rpc_request_batch(substrate_instance, [('method1', ['param1']), ('method2', ['param2'])])
            ['result1', 'result2', ...]
        """

        chunk_results: list[Any] = []
        # smaller_requests = self._make_request_smaller(batch_requests)
        request_id = 0
        with ThreadPoolExecutor() as executor:
            futures: list[Future[list[str | dict[Any, Any]]]] = []
            for chunk in [batch_requests]:
                request_ids: list[int] = []
                batch_payload: list[Any] = []
                for method, params in chunk:
                    request_id += 1
                    request_ids.append(request_id)
                    batch_payload.append(
                        {
                            "jsonrpc": "2.0",
                            "method": method,
                            "params": params,
                            "id": request_id,
                        }
                    )

                futures.append(
                    executor.submit(
                        self._send_batch,
                        batch_payload=batch_payload,
                        request_ids=request_ids,
                        extract_result=extract_result,
                    )
                )
            for future in futures:
                resul = future.result()
                chunk_results.append(resul)
        return chunk_results

    def _rpc_request_batch_chunked(
        self, chunk_requests: list[Chunk], extract_result: bool = True
    ):
        """
        Sends batch requests to the substrate node using multiple threads and collects the results.

        Args:
            substrate: An instance of the substrate interface.
            batch_requests : A list of requests to be sent in batches.
            max_size: Maximum size of each batch in bytes.
            extract_result: Whether to extract the result from the response message.

        Returns:
            A list of results from the batch requests.

        Example:
            >>> _rpc_request_batch(substrate_instance, [('method1', ['param1']), ('method2', ['param2'])])
            ['result1', 'result2', ...]
        """

        def split_chunks(chunk: Chunk, chunk_info: list[Chunk], chunk_info_idx: int):
            manhattam_chunks: list[tuple[Any, Any]] = []
            mutaded_chunk_info = deepcopy(chunk_info)
            max_n_keys = 35000
            for query in chunk.batch_requests:
                result_keys = query[1][0]
                keys_amount = len(result_keys)
                if keys_amount > max_n_keys:
                    mutaded_chunk_info.pop(chunk_info_idx)
                    for i in range(0, keys_amount, max_n_keys):
                        new_chunk = deepcopy(chunk)
                        splitted_keys = result_keys[i : i + max_n_keys]
                        splitted_query = deepcopy(query)
                        splitted_query[1][0] = splitted_keys
                        new_chunk.batch_requests = [splitted_query]
                        manhattam_chunks.append(splitted_query)
                        mutaded_chunk_info.insert(chunk_info_idx, new_chunk)
                else:
                    manhattam_chunks.append(query)
            return manhattam_chunks, mutaded_chunk_info

        assert len(chunk_requests) > 0
        mutated_chunk_info: list[Chunk] = []
        chunk_results: list[Any] = []
        # smaller_requests = self._make_request_smaller(batch_requests)
        request_id = 0

        with ThreadPoolExecutor() as executor:
            futures: list[Future[list[str | dict[Any, Any]]]] = []
            for idx, macro_chunk in enumerate(chunk_requests):
                _, mutated_chunk_info = split_chunks(macro_chunk, chunk_requests, idx)
            for chunk in mutated_chunk_info:
                request_ids: list[int] = []
                batch_payload: list[Any] = []
                for method, params in chunk.batch_requests:
                    # for method, params in micro_chunk:
                    request_id += 1
                    request_ids.append(request_id)
                    batch_payload.append(
                        {
                            "jsonrpc": "2.0",
                            "method": method,
                            "params": params,
                            "id": request_id,
                        }
                    )
                futures.append(
                    executor.submit(
                        self._send_batch,
                        batch_payload=batch_payload,
                        request_ids=request_ids,
                        extract_result=extract_result,
                    )
                )
            for future in futures:
                resul = future.result()
                chunk_results.append(resul)
        return chunk_results, mutated_chunk_info

    def _decode_response(
        self,
        response: list[str],
        function_parameters: list[tuple[Any, Any, Any, Any, str]],
        prefix_list: list[Any],
        block_hash: str,
    ) -> dict[str, dict[Any, Any]]:
        """
        Decodes a response from the substrate interface and organizes the data into a dictionary.

        Args:
            response: A list of encoded responses from a substrate query.
            function_parameters: A list of tuples containing the parameters for each storage function.
            last_keys: A list of the last keys used in the substrate query.
            prefix_list: A list of prefixes used in the substrate query.
            substrate: An instance of the SubstrateInterface class.
            block_hash: The hash of the block to be queried.

        Returns:
            A dictionary where each key is a storage function name and the value is another dictionary.
            This inner dictionary's key is the decoded key from the response and the value is the corresponding decoded value.

        Raises:
            ValueError: If an unsupported hash type is encountered in the `concat_hash_len` function.

        Example:
            >>> _decode_response(
                    response=[...],
                    function_parameters=[...],
                    last_keys=[...],
                    prefix_list=[...],
                    substrate=substrate_instance,
                    block_hash="0x123..."
                )
            {'storage_function_name': {decoded_key: decoded_value, ...}, ...}
        """

        def concat_hash_len(key_hasher: str) -> int:
            """
            Determines the length of the hash based on the given key hasher type.

            Args:
                key_hasher: The type of key hasher.

            Returns:
                The length of the hash corresponding to the given key hasher type.

            Raises:
                ValueError: If the key hasher type is not supported.

            Example:
                >>> concat_hash_len("Blake2_128Concat")
                16
            """

            if key_hasher == "Blake2_128Concat":
                return 16
            elif key_hasher == "Twox64Concat":
                return 8
            elif key_hasher == "Identity":
                return 0
            else:
                raise ValueError("Unsupported hash type")

        assert len(response) == len(function_parameters) == len(prefix_list)
        result_dict: dict[str, dict[Any, Any]] = {}
        for res, fun_params_tuple, prefix in zip(
            response, function_parameters, prefix_list
        ):
            if not res:
                continue
            res = res[0]
            changes = res["changes"]  # type: ignore
            value_type, param_types, key_hashers, params, storage_function = (
                fun_params_tuple
            )
            with self.get_conn(init=True) as substrate:
                for item in changes:
                    # Determine type string
                    key_type_string: list[Any] = []
                    for n in range(len(params), len(param_types)):
                        key_type_string.append(
                            f"[u8; {concat_hash_len(key_hashers[n])}]"
                        )
                        key_type_string.append(param_types[n])

                    item_key_obj = substrate.decode_scale(  # type: ignore
                        type_string=f"({', '.join(key_type_string)})",
                        scale_bytes="0x" + item[0][len(prefix) :],
                        return_scale_obj=True,
                        block_hash=block_hash,
                    )
                    # strip key_hashers to use as item key
                    if len(param_types) - len(params) == 1:
                        item_key = item_key_obj.value_object[1]  # type: ignore
                    else:
                        item_key = tuple(  # type: ignore
                            item_key_obj.value_object[key + 1]  # type: ignore
                            for key in range(  # type: ignore
                                len(params), len(param_types) + 1, 2
                            )
                        )

                    item_value = substrate.decode_scale(  # type: ignore
                        type_string=value_type,
                        scale_bytes=item[1],
                        return_scale_obj=True,
                        block_hash=block_hash,
                    )
                    result_dict.setdefault(storage_function, {})

                    result_dict[storage_function][item_key.value] = item_value.value  #  type: ignore

        return result_dict

    def query_batch(
        self, functions: dict[str, list[tuple[str, list[Any]]]]
    ) -> dict[str, str]:
        """
        Executes batch queries on a substrate and returns results in a dictionary format.

        Args:
            substrate: An instance of SubstrateInterface to interact with the substrate.
            functions (dict[str, list[query_call]]): A dictionary mapping module names to lists of query calls (function name and parameters).

        Returns:
            A dictionary where keys are storage function names and values are the query results.

        Raises:
            Exception: If no result is found from the batch queries.

        Example:
            >>> query_batch(substrate_instance, {'module_name': [('function_name', ['param1', 'param2'])]})
            {'function_name': 'query_result', ...}
        """

        result = None
        with self.get_conn(init=True) as substrate:
            for module, queries in functions.items():
                storage_keys: list[Any] = []
                for fn, params in queries:
                    storage_function = substrate.create_storage_key(  # type: ignore
                        pallet=module, storage_function=fn, params=params
                    )
                    storage_keys.append(storage_function)

                block_hash = substrate.get_block_hash()
                responses: list[Any] = substrate.query_multi(  # type: ignore
                    storage_keys=storage_keys, block_hash=block_hash
                )

                result: dict[str, str] | None = {}

                for item in responses:
                    fun = item[0]
                    query = item[1]
                    storage_fun = fun.storage_function
                    result[storage_fun] = query.value

            if result is None:
                raise Exception("No result")

        return result

    def query_batch_map(
        self,
        functions: dict[str, list[tuple[str, list[Any]]]],
        block_hash: str | None = None,
    ) -> dict[str, dict[Any, Any]]:
        """
        Queries multiple storage functions using a map batch approach and returns the combined result.

        Args:
            substrate: An instance of SubstrateInterface for substrate interaction.
            functions (dict[str, list[query_call]]): A dictionary mapping module names to lists of query calls.

        Returns:
            The combined result of the map batch query.

        Example:
            >>> query_batch_map(substrate_instance, {'module_name': [('function_name', ['param1', 'param2'])]})
            # Returns the combined result of the map batch query
        """
        multi_result: dict[str, dict[Any, Any]] = {}

        def recursive_update(
            d: dict[str, dict[T1, T2] | dict[str, Any]],
            u: Mapping[str, dict[Any, Any] | str],
        ) -> dict[str, dict[T1, T2]]:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = recursive_update(d.get(k, {}), v)  # type: ignore
                else:
                    d[k] = v  # type: ignore
            return d  # type: ignore

        def get_page():
            send, prefix_list = self._get_storage_keys(storage, queries, block_hash)
            with self.get_conn(init=True) as substrate:
                function_parameters = self._get_lists(storage, queries, substrate)
            responses = self._rpc_request_batch(send)
            # assumption because send is just the storage_function keys
            # so it should always be really small regardless of the amount of queries
            assert len(responses) == 1
            res = responses[0]
            built_payload: list[tuple[str, list[Any]]] = []
            for result_keys in res:
                built_payload.append(
                    ("state_queryStorageAt", [result_keys, block_hash])
                )
            _, chunks_info = self._make_request_smaller(
                built_payload, prefix_list, function_parameters
            )
            chunks_response, chunks_info = self._rpc_request_batch_chunked(chunks_info)
            return chunks_response, chunks_info

        if not block_hash:
            with self.get_conn(init=True) as substrate:
                block_hash = substrate.get_block_hash()
        for storage, queries in functions.items():
            chunks, chunks_info = get_page()
            # if this doesn't happen something is wrong on the code
            # and we won't be able to decode the data properly
            assert len(chunks) == len(chunks_info)
            for chunk_info, response in zip(chunks_info, chunks):
                storage_result = self._decode_response(
                    response, chunk_info.fun_params, chunk_info.prefix_list, block_hash
                )
                multi_result = recursive_update(multi_result, storage_result)

        return multi_result

    def query(
        self,
        name: str,
        params: list[Any] = [],
        module: str = "SubspaceModule",
    ) -> Any:
        """
        Queries a storage function on the network.

        Sends a query to the network and retrieves data from a
        specified storage function.

        Args:
            name: The name of the storage function to query.
            params: The parameters to pass to the storage function.
            module: The module where the storage function is located.

        Returns:
            The result of the query from the network.

        Raises:
            NetworkQueryError: If the query fails or is invalid.
        """

        result = self.query_batch({module: [(name, params)]})

        return result[name]

    def query_map(
        self,
        name: str,
        params: list[Any] = [],
        module: str = "SubspaceModule",
        extract_value: bool = True,
    ) -> dict[Any, Any]:
        """
        Queries a storage map from a network node.

        Args:
            name: The name of the storage map to query.
            params: A list of parameters for the query.
            module: The module in which the storage map is located.

        Returns:
            A dictionary representing the key-value pairs
              retrieved from the storage map.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        result = self.query_batch_map({module: [(name, params)]})

        if extract_value:
            return {k.value: v.value for k, v in result}  # type: ignore

        return result

    def compose_call(
        self,
        fn: str,
        params: dict[str, Any],
        key: Keypair,
        module: str = "SubspaceModule",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool | None = None,
        sudo: bool = False,
    ) -> ExtrinsicReceipt:
        """
        Composes and submits a call to the network node.

        Composes and signs a call with the provided keypair, and submits it to
        the network. The call can be a standard extrinsic or a sudo extrinsic if
        elevated permissions are required. The method can optionally wait for
        the call's inclusion in a block and/or its finalization.

        Args:
            fn: The function name to call on the network.
            params: A dictionary of parameters for the call.
            key: The keypair for signing the extrinsic.
            module: The module containing the function.
            wait_for_inclusion: Wait for the call's inclusion in a block.
            wait_for_finalization: Wait for the transaction's finalization.
            sudo: Execute the call as a sudo (superuser) operation.

        Returns:
            The receipt of the submitted extrinsic, if
              `wait_for_inclusion` is True. Otherwise, returns a string
              identifier of the extrinsic.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        with self.get_conn() as substrate:
            if wait_for_finalization is None:
                wait_for_finalization = self.wait_for_finalization

            call = substrate.compose_call(  # type: ignore
                call_module=module, call_function=fn, call_params=params
            )
            if sudo:
                call = substrate.compose_call(  # type: ignore
                    call_module="Sudo",
                    call_function="sudo",
                    call_params={
                        "call": call.value,  # type: ignore
                    },
                )

            extrinsic = substrate.create_signed_extrinsic(  # type: ignore
                call=call,
                keypair=key,  # type: ignore
            )  # type: ignore
            response = substrate.submit_extrinsic(
                extrinsic=extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
        if wait_for_inclusion:
            if not response.is_success:
                raise ChainTransactionError(
                    response.error_message,
                    response,  # type: ignore
                )

        return response

    def compose_call_multisig(
        self,
        fn: str,
        params: dict[str, Any],
        key: Keypair,
        signatories: list[Ss58Address],
        threshold: int,
        module: str = "SubspaceModule",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool | None = None,
        sudo: bool = False,
        era: dict[str, int] | None = None,
    ) -> ExtrinsicReceipt:
        """
        Composes and submits a multisignature call to the network node.

        This method allows the composition and submission of a call that
        requires multiple signatures for execution, known as a multisignature
        call. It supports specifying signatories, a threshold of signatures for
        the call's execution, and an optional era for the call's mortality. The
        call can be a standard extrinsic, a sudo extrinsic for elevated
        permissions, or a multisig extrinsic if multiple signatures are
        required. Optionally, the method can wait for the call's inclusion in a
        block and/or its finalization. Make sure to pass all keys,
        that are part of the multisignature.

        Args:
            fn: The function name to call on the network. params: A dictionary
            of parameters for the call. key: The keypair for signing the
            extrinsic. signatories: List of SS58 addresses of the signatories.
            Include ALL KEYS that are part of the multisig. threshold: The
            minimum number of signatories required to execute the extrinsic.
            module: The module containing the function to call.
            wait_for_inclusion: Whether to wait for the call's inclusion in a
            block. wait_for_finalization: Whether to wait for the transaction's
            finalization. sudo: Execute the call as a sudo (superuser)
            operation. era: Specifies the call's mortality in terms of blocks in
            the format
                {'period': amount_blocks}. If omitted, the extrinsic is
                immortal.

        Returns:
            The receipt of the submitted extrinsic if `wait_for_inclusion` is
            True. Otherwise, returns a string identifier of the extrinsic.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        # getting the call ready
        with self.get_conn() as substrate:
            if wait_for_finalization is None:
                wait_for_finalization = self.wait_for_finalization

            # prepares the `GenericCall` object
            call = substrate.compose_call(  # type: ignore
                call_module=module, call_function=fn, call_params=params
            )
            if sudo:
                call = substrate.compose_call(  # type: ignore
                    call_module="Sudo",
                    call_function="sudo",
                    call_params={
                        "call": call.value,  # type: ignore
                    },
                )

            # modify the rpc methods at runtime, to allow for correct payment
            # fee calculation parity has a bug in this version,
            # where the method has to be removed
            rpc_methods = substrate.config.get("rpc_methods")  # type: ignore

            if "state_call" in rpc_methods:  # type: ignore
                rpc_methods.remove("state_call")  # type: ignore

            # create the multisig account
            multisig_acc = substrate.generate_multisig_account(  # type: ignore
                signatories, threshold
            )

            # send the multisig extrinsic
            extrinsic = substrate.create_multisig_extrinsic(  # type: ignore
                call=call,
                keypair=key,
                multisig_account=multisig_acc,  # type: ignore
                era=era,  # type: ignore
            )  # type: ignore

            response = substrate.submit_extrinsic(
                extrinsic=extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        if wait_for_inclusion:
            if not response.is_success:
                raise ChainTransactionError(
                    response.error_message,
                    response,  # type: ignore
                )

        return response

    def transfer(
        self,
        key: Keypair,
        amount: int,
        dest: Ss58Address,
    ) -> ExtrinsicReceipt:
        """
        Transfers a specified amount of tokens from the signer's account to the
        specified account.

        Args:
            key: The keypair associated with the sender's account.
            amount: The amount to transfer, in nanotokens.
            dest: The SS58 address of the recipient.

        Returns:
            A receipt of the transaction.

        Raises:
            InsufficientBalanceError: If the sender's account does not have
              enough balance.
            ChainTransactionError: If the transaction fails.
        """

        params = {"dest": dest, "value": amount}

        return self.compose_call(
            module="Balances", fn="transfer", params=params, key=key
        )

    def transfer_multiple(
        self,
        key: Keypair,
        destinations: list[Ss58Address],
        amounts: list[int],
        netuid: str | int = 0,
    ) -> ExtrinsicReceipt:
        """
        Transfers specified amounts of tokens from the signer's account to
        multiple target accounts.

        The `destinations` and `amounts` lists must be of the same length.

        Args:
            key: The keypair associated with the sender's account.
            destinations: A list of SS58 addresses of the recipients.
            amounts: Amount to transfer to each recipient, in nanotokens.
            netuid: The network identifier.

        Returns:
            A receipt of the transaction.

        Raises:
            InsufficientBalanceError: If the sender's account does not have
              enough balance for all transfers.
            ChainTransactionError: If the transaction fails.
        """

        assert len(destinations) == len(amounts)

        # extract existential deposit from amounts
        existential_deposit = self.get_existential_deposit()
        amounts = [a - existential_deposit for a in amounts]

        params = {
            "netuid": netuid,
            "destinations": destinations,
            "amounts": amounts,
        }

        return self.compose_call(
            module="SubspaceModule", fn="transfer_multiple", params=params, key=key
        )

    def stake(
        self,
        key: Keypair,
        amount: int,
        dest: Ss58Address,
        netuid: int = 0,
    ) -> ExtrinsicReceipt:
        """
        Stakes the specified amount of tokens to a module key address.

        Args:
            key: The keypair associated with the staker's account.
            amount: The amount of tokens to stake, in nanotokens.
            dest: The SS58 address of the module key to stake to.
            netuid: The network identifier.

        Returns:
            A receipt of the staking transaction.

        Raises:
            InsufficientBalanceError: If the staker's account does not have
              enough balance.
            ChainTransactionError: If the transaction fails.
        """

        params = {"amount": amount, "netuid": netuid, "module_key": dest}

        return self.compose_call(fn="add_stake", params=params, key=key)

    def unstake(
        self,
        key: Keypair,
        amount: int,
        dest: Ss58Address,
        netuid: int = 0,
    ) -> ExtrinsicReceipt:
        """
        Unstakes the specified amount of tokens from a module key address.

        Args:
            key: The keypair associated with the unstaker's account.
            amount: The amount of tokens to unstake, in nanotokens.
            dest: The SS58 address of the module key to unstake from.
            netuid: The network identifier.

        Returns:
            A receipt of the unstaking transaction.

        Raises:
            InsufficientStakeError: If the staked key does not have enough
              staked tokens by the signer key.
            ChainTransactionError: If the transaction fails.
        """

        params = {"amount": amount, "netuid": netuid, "module_key": dest}
        return self.compose_call(fn="remove_stake", params=params, key=key)

    def update_module(
        self,
        key: Keypair,
        name: str,
        address: str,
        metadata: str | None = None,
        delegation_fee: int = 20,
        netuid: int = 0,
    ) -> ExtrinsicReceipt:
        """
        Updates the parameters of a registered module.

        The delegation fee must be an integer between 0 and 100.

        Args:
            key: The keypair associated with the module's account.
            name: The new name for the module. If None, the name is not updated.
            address: The new address for the module.
                If None, the address is not updated.
            delegation_fee: The new delegation fee for the module,
                between 0 and 100.
            netuid: The network identifier.

        Returns:
            A receipt of the module update transaction.

        Raises:
            InvalidParameterError: If the provided parameters are invalid.
            ChainTransactionError: If the transaction fails.
        """

        assert isinstance(delegation_fee, int)

        params = {
            "netuid": netuid,
            "name": name,
            "address": address,
            "delegation_fee": delegation_fee,
            "metadata": metadata,
        }

        response = self.compose_call("update_module", params=params, key=key)

        return response

    def register_module(
        self,
        key: Keypair,
        name: str,
        address: str | None = None,
        subnet: str = "commune",
        min_stake: int | None = None,
        metadata: str | None = None,
    ) -> ExtrinsicReceipt:
        """
        Registers a new module in the network.

        Args:
            key: The keypair used for registering the module.
            name: The name of the module. If None, a default or previously
                set name is used. # How does this work?
            address: The address of the module. If None, a default or
                previously set address is used. # How does this work?
            subnet: The network subnet to register the module in.
            min_stake: The minimum stake required for the module, in nanotokens.
                If None, a default value is used.

        Returns:
            A receipt of the registration transaction.

        Raises:
            InvalidParameterError: If the provided parameters are invalid.
            ChainTransactionError: If the transaction fails.
        """

        stake = self.get_min_stake() if min_stake is None else min_stake

        key_addr = key.ss58_address

        params = {
            "network": subnet,
            "address": address,
            "name": name,
            "stake": stake,
            "module_key": key_addr,
            "metadata": metadata,
        }

        response = self.compose_call("register", params=params, key=key)
        return response

    def vote(
        self,
        key: Keypair,
        uids: list[int],
        weights: list[int],
        netuid: int = 0,
    ) -> ExtrinsicReceipt:
        """
        Casts votes on a list of module UIDs with corresponding weights.

        The length of the UIDs list and the weights list should be the same.
        Each weight corresponds to the UID at the same index.

        Args:
            key: The keypair used for signing the vote transaction.
            uids: A list of module UIDs to vote on.
            weights: A list of weights corresponding to each UID.
            netuid: The network identifier.

        Returns:
            A receipt of the voting transaction.

        Raises:
            InvalidParameterError: If the lengths of UIDs and weights lists
                do not match.
            ChainTransactionError: If the transaction fails.
        """

        assert len(uids) == len(weights)

        params = {
            "uids": uids,
            "weights": weights,
            "netuid": netuid,
        }

        response = self.compose_call("set_weights", params=params, key=key)

        return response

    def update_subnet(
        self,
        key: Keypair,
        params: SubnetParams,
        netuid: int = 0,
    ) -> ExtrinsicReceipt:
        """
        Update a subnet's configuration.

        It requires the founder key for authorization.

        Args:
            key: The founder keypair of the subnet.
            params: The new parameters for the subnet.
            netuid: The network identifier.

        Returns:
            A receipt of the subnet update transaction.

        Raises:
            AuthorizationError: If the key is not authorized.
            ChainTransactionError: If the transaction fails.
        """

        general_params = dict(params)
        general_params["netuid"] = netuid

        response = self.compose_call(
            fn="update_subnet",
            params=general_params,
            key=key,
        )

        return response

    def transfer_stake(
        self,
        key: Keypair,
        amount: int,
        from_module_key: Ss58Address,
        dest_module_address: Ss58Address,
        netuid: int = 0,
    ) -> ExtrinsicReceipt:
        """
        Realocate staked tokens from one staked module to another module.

        Args:
            key: The keypair associated with the account that is delegating the tokens.
            amount: The amount of staked tokens to transfer, in nanotokens.
            from_module_key: The SS58 address of the module you want to transfer from (currently delegated by the key).
            dest_module_address: The SS58 address of the destination (newly delegated key).
            netuid: The network identifier.

        Returns:
            A receipt of the stake transfer transaction.

        Raises:
            InsufficientStakeError: If the source module key does not have
            enough staked tokens. ChainTransactionError: If the transaction
            fails.
        """

        amount = amount - self.get_existential_deposit()

        params = {
            "amount": amount,
            "netuid": netuid,
            "module_key": from_module_key,
            "new_module_key": dest_module_address,
        }

        response = self.compose_call("transfer_stake", key=key, params=params)

        return response

    def multiunstake(
        self,
        key: Keypair,
        keys: list[Ss58Address],
        amounts: list[int],
        netuid: int = 0,
    ) -> ExtrinsicReceipt:
        """
        Unstakes tokens from multiple module keys.

        And the lists `keys` and `amounts` must be of the same length. Each
        amount corresponds to the module key at the same index.

        Args:
            key: The keypair associated with the unstaker's account.
            keys: A list of SS58 addresses of the module keys to unstake from.
            amounts: A list of amounts to unstake from each module key,
              in nanotokens.
            netuid: The network identifier.

        Returns:
            A receipt of the multi-unstaking transaction.

        Raises:
            MismatchedLengthError: If the lengths of keys and amounts lists do
            not match. InsufficientStakeError: If any of the module keys do not
            have enough staked tokens. ChainTransactionError: If the transaction
            fails.
        """

        assert len(keys) == len(amounts)

        params = {"netuid": netuid, "module_keys": keys, "amounts": amounts}

        response = self.compose_call("remove_stake_multiple", params=params, key=key)

        return response

    def multistake(
        self,
        key: Keypair,
        keys: list[Ss58Address],
        amounts: list[int],
        netuid: int = 0,
    ) -> ExtrinsicReceipt:
        """
        Stakes tokens to multiple module keys.

        The lengths of the `keys` and `amounts` lists must be the same. Each
        amount corresponds to the module key at the same index.

        Args:
            key: The keypair associated with the staker's account.
            keys: A list of SS58 addresses of the module keys to stake to.
            amounts: A list of amounts to stake to each module key,
                in nanotokens.
            netuid: The network identifier.

        Returns:
            A receipt of the multi-staking transaction.

        Raises:
            MismatchedLengthError: If the lengths of keys and amounts lists
                do not match.
            ChainTransactionError: If the transaction fails.
        """

        assert len(keys) == len(amounts)

        params = {
            "module_keys": keys,
            "amounts": amounts,
            "netuid": netuid,
        }

        response = self.compose_call("add_stake_multiple", params=params, key=key)

        return response

    def add_profit_shares(
        self,
        key: Keypair,
        keys: list[Ss58Address],
        shares: list[int],
    ) -> ExtrinsicReceipt:
        """
        Allocates profit shares to multiple keys.

        The lists `keys` and `shares` must be of the same length,
        with each share amount corresponding to the key at the same index.

        Args:
            key: The keypair associated with the account
                distributing the shares.
            keys: A list of SS58 addresses to allocate shares to.
            shares: A list of share amounts to allocate to each key,
                in nanotokens.

        Returns:
            A receipt of the profit sharing transaction.

        Raises:
            MismatchedLengthError: If the lengths of keys and shares
                lists do not match.
            ChainTransactionError: If the transaction fails.
        """

        assert len(keys) == len(shares)

        params = {"keys": keys, "shares": shares}

        response = self.compose_call("add_profit_shares", params=params, key=key)

        return response

    def add_subnet_proposal(
        self, key: Keypair, params: SubnetParams, netuid: int = 0
    ) -> ExtrinsicReceipt:
        """
        Submits a proposal for creating or modifying a subnet within the
        network.

        The proposal includes various parameters like the name, founder, share
        allocations, and other subnet-specific settings.

        Args:
            key: The keypair used for signing the proposal transaction.
            params: The parameters for the subnet proposal.
            netuid: The network identifier.

        Returns:
            A receipt of the subnet proposal transaction.

        Raises:
            InvalidParameterError: If the provided subnet
                parameters are invalid.
            ChainTransactionError: If the transaction fails.
        """

        general_params = dict(params)
        general_params["netuid"] = netuid

        response = self.compose_call(
            fn="add_subnet_proposal",
            params=general_params,
            key=key,
        )

        return response

    def add_custom_proposal(
        self,
        key: Keypair,
        cid: str,
    ) -> ExtrinsicReceipt:
        params = {"data": cid}

        response = self.compose_call(fn="add_custom_proposal", params=params, key=key)
        return response

    def add_custom_subnet_proposal(
        self,
        key: Keypair,
        cid: str,
        netuid: int = 0,
    ) -> ExtrinsicReceipt:
        """
        Submits a proposal for creating or modifying a custom subnet within the
        network.

        The proposal includes various parameters like the name, founder, share
        allocations, and other subnet-specific settings.

        Args:
            key: The keypair used for signing the proposal transaction.
            params: The parameters for the subnet proposal.
            netuid: The network identifier.

        Returns:
            A receipt of the subnet proposal transaction.
        """

        params = {
            "data": cid,
            "netuid": netuid,
        }

        response = self.compose_call(
            fn="add_custom_subnet_proposal",
            params=params,
            key=key,
        )

        return response

    def add_global_proposal(
        self,
        key: Keypair,
        params: NetworkParams,
    ) -> ExtrinsicReceipt:
        """
        Submits a proposal for altering the global network parameters.

        Allows for the submission of a proposal to
        change various global parameters
        of the network, such as emission rates, rate limits, and voting
        thresholds. It is used to
        suggest changes that affect the entire network's operation.

        Args:
            key: The keypair used for signing the proposal transaction.
            params: A dictionary containing global network parameters
                    like maximum allowed subnets, modules,
                    transaction rate limits, and others.

        Returns:
            A receipt of the global proposal transaction.

        Raises:
            InvalidParameterError: If the provided network
                parameters are invalid.
            ChainTransactionError: If the transaction fails.
        """

        general_params = vars(params)
        response = self.compose_call(
            fn="add_global_proposal",
            params=general_params,
            key=key,
        )

        return response

    def vote_on_proposal(
        self,
        key: Keypair,
        proposal_id: int,
        agree: bool,
    ) -> ExtrinsicReceipt:
        """
        Casts a vote on a specified proposal within the network.

        Args:
            key: The keypair used for signing the vote transaction.
            proposal_id: The unique identifier of the proposal to vote on.

        Returns:
            A receipt of the voting transaction in nanotokens.

        Raises:
            InvalidProposalIDError: If the provided proposal ID does not
                exist or is invalid.
            ChainTransactionError: If the transaction fails.
        """

        params = {"proposal_id": proposal_id, "agree": agree}

        response = self.compose_call("vote_proposal", key=key, params=params)

        return response

    def unvote_on_proposal(
        self,
        key: Keypair,
        proposal_id: int,
    ) -> ExtrinsicReceipt:
        """
        Retracts a previously cast vote on a specified proposal.

        Args:
            key: The keypair used for signing the unvote transaction.
            proposal_id: The unique identifier of the proposal to withdraw the
                vote from.

        Returns:
            A receipt of the unvoting transaction in nanotokens.

        Raises:
            InvalidProposalIDError: If the provided proposal ID does not
                exist or is invalid.
            ChainTransactionError: If the transaction fails to be processed, or
                if there was no prior vote to retract.
        """

        params = {"proposal_id": proposal_id}

        response = self.compose_call("unvote_proposal", key=key, params=params)

        return response

    def add_dao_application(
        self, key: Keypair, application_key: Ss58Address, data: str
    ) -> ExtrinsicReceipt:
        """
        Submits a new application to the general subnet DAO.

        Args:
            key: The keypair used for signing the application transaction.
            application_key: The SS58 address of the application key.
            data: The data associated with the application.

        Returns:
            A receipt of the application transaction.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        params = {"application_key": application_key, "data": data}

        response = self.compose_call("add_dao_application", key=key, params=params)

        return response

    def query_map_curator_applications(self) -> dict[str, dict[str, str]]:
        query_result = self.query_map(
            "CuratorApplications", params=[], extract_value=False
        )
        applications = query_result.get("CuratorApplications", {})
        return applications

    def query_map_proposals(
        self, extract_value: bool = False
    ) -> dict[int, dict[str, Any]]:
        """
        Retrieves a mappping of proposals from the network.

        Queries the network and returns a mapping of proposal IDs to
        their respective parameters.

        Returns:
            A dictionary mapping proposal IDs
            to dictionaries of their parameters.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("Proposals", extract_value=extract_value)["Proposals"]

    def query_map_weights(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[int, list[int]]:
        """
        Retrieves a mapping of weights for keys on the network.

        Queries the network and returns a mapping of key UIDs to
        their respective weights.

        Args:
            netuid: The network UID from which to get the weights.

        Returns:
            A dictionary mapping key UIDs to lists of their weights.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("Weights", [netuid], extract_value=extract_value)[
            "Weights"
        ]

    def query_map_key(
        self,
        netuid: int = 0,
        extract_value: bool = False,
    ) -> dict[int, Ss58Address]:
        """
        Retrieves a map of keys from the network.

        Fetches a mapping of key UIDs to their associated
        addresses on the network.
        The query can be targeted at a specific network UID if required.

        Args:
            netuid: The network UID from which to get the keys.

        Returns:
            A dictionary mapping key UIDs to their addresses.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """
        return self.query_map("Keys", [netuid], extract_value=extract_value)["Keys"]

    def query_map_address(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[int, str]:
        """
        Retrieves a map of key addresses from the network.

        Queries the network for a mapping of key UIDs to their addresses.

        Args:
            netuid: The network UID from which to get the addresses.

        Returns:
            A dictionary mapping key UIDs to their addresses.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("Address", [netuid], extract_value=extract_value)[
            "Address"
        ]

    def query_map_emission(self, extract_value: bool = False) -> dict[int, list[int]]:
        """
        Retrieves a map of emissions for keys on the network.

        Queries the network to get a mapping of
        key UIDs to their emission values.

        Returns:
            A dictionary mapping key UIDs to lists of their emission values.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("Emission", extract_value=extract_value)["Emission"]

    def query_map_incentive(self, extract_value: bool = False) -> dict[int, list[int]]:
        """
        Retrieves a mapping of incentives for keys on the network.

        Queries the network and returns a mapping of key UIDs to
        their respective incentive values.

        Returns:
            A dictionary mapping key UIDs to lists of their incentive values.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("Incentive", extract_value=extract_value)["Incentive"]

    def query_map_dividend(self, extract_value: bool = False) -> dict[int, list[int]]:
        """
        Retrieves a mapping of dividends for keys on the network.

        Queries the network for a mapping of key UIDs to
        their dividend values.

        Returns:
            A dictionary mapping key UIDs to lists of their dividend values.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("Dividends", extract_value=extract_value)["Dividends"]

    def query_map_regblock(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[int, int]:
        """
        Retrieves a mapping of registration blocks for keys on the network.

        Queries the network for a mapping of key UIDs to
        the blocks where they were registered.

        Args:
            netuid: The network UID from which to get the registration blocks.

        Returns:
            A dictionary mapping key UIDs to their registration blocks.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map(
            "RegistrationBlock", [netuid], extract_value=extract_value
        )["RegistrationBlock"]

    def query_map_lastupdate(self, extract_value: bool = False) -> dict[int, list[int]]:
        """
        Retrieves a mapping of the last update times for keys on the network.

        Queries the network for a mapping of key UIDs to their last update times.

        Returns:
            A dictionary mapping key UIDs to lists of their last update times.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("LastUpdate", extract_value=extract_value)["LastUpdate"]

    def query_map_total_stake(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of total stakes for keys on the network.

        Queries the network for a mapping of key UIDs to their total stake amounts.

        Returns:
            A dictionary mapping key UIDs to their total stake amounts.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("TotalStake", extract_value=extract_value)["TotalStake"]

    def query_map_stakefrom(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[str, list[tuple[str, int]]]:
        """
        Retrieves a mapping of stakes from various sources for keys on the network.

        Queries the network to obtain a mapping of key addresses to the sources
        and amounts of stakes they have received.

        Args:
            netuid: The network UID from which to get the stakes.

        Returns:
            A dictionary mapping key addresses to lists of tuples
            (module_key_address, amount).

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("StakeFrom", [netuid], extract_value=extract_value)[
            "StakeFrom"
        ]

    def query_map_staketo(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[str, list[tuple[str, int]]]:
        """
        Retrieves a mapping of stakes to destinations for keys on the network.

        Queries the network for a mapping of key addresses to the destinations
        and amounts of stakes they have made.

        Args:
            netuid: The network UID from which to get the stakes.

        Returns:
            A dictionary mapping key addresses to lists of tuples
            (module_key_address, amount).

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("StakeTo", [netuid], extract_value=extract_value)[
            "StakeTo"
        ]

    def query_map_stake(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[str, int]:
        """
        Retrieves a mapping of stakes for keys on the network.

        Queries the network and returns a mapping of key addresses to their
        respective delegated staked balances amounts.
        The query can be targeted at a specific network UID if required.

        Args:
            netuid: The network UID from which to get the stakes.

        Returns:
            A dictionary mapping key addresses to their stake amounts.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("Stake", [netuid], extract_value=extract_value)["Stake"]

    def query_map_delegationfee(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[str, int]:
        """
        Retrieves a mapping of delegation fees for keys on the network.

        Queries the network to obtain a mapping of key addresses to their
        respective delegation fees.

        Args:
            netuid: The network UID to filter the delegation fees.

        Returns:
            A dictionary mapping key addresses to their delegation fees.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("DelegationFee", [netuid], extract_value=extract_value)[
            "DelegationFee"
        ]

    def query_map_tempo(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of tempo settings for the network.

        Queries the network to obtain the tempo (rate of reward distributions)
        settings for various network subnets.

        Returns:
            A dictionary mapping network UIDs to their tempo settings.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("Tempo", extract_value=extract_value)["Tempo"]

    def query_map_immunity_period(self, extract_value: bool) -> dict[int, int]:
        """
        Retrieves a mapping of immunity periods for the network.

        Queries the network for the immunity period settings,
        which represent the time duration during which modules
        can not get deregistered.

        Returns:
            A dictionary mapping network UIDs to their immunity period settings.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("ImmunityPeriod", extract_value=extract_value)[
            "ImmunityPeriod"
        ]

    def query_map_min_allowed_weights(
        self, extract_value: bool = False
    ) -> dict[int, int]:
        """
        Retrieves a mapping of minimum allowed weights for the network.

        Queries the network to obtain the minimum allowed weights,
        which are the lowest permissible weight values that can be set by
        validators.

        Returns:
            A dictionary mapping network UIDs to
            their minimum allowed weight values.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("MinAllowedWeights", extract_value=extract_value)[
            "MinAllowedWeights"
        ]

    def query_map_max_allowed_weights(
        self, extract_value: bool = False
    ) -> dict[int, int]:
        """
        Retrieves a mapping of maximum allowed weights for the network.

        Queries the network for the maximum allowed weights,
        which are the highest permissible
        weight values that can be set by validators.

        Returns:
            A dictionary mapping network UIDs to
            their maximum allowed weight values.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("MaxAllowedWeights", extract_value=extract_value)[
            "MaxAllowedWeights"
        ]

    def query_map_max_allowed_uids(self, extract_value: bool = False) -> dict[int, int]:
        """
        Queries the network for the maximum number of allowed user IDs (UIDs)
        for each network subnet.

        Fetches a mapping of network subnets to their respective
        limits on the number of user IDs that can be created or used.

        Returns:
            A dictionary mapping network UIDs (unique identifiers) to their
            maximum allowed number of UIDs.
            Each entry represents a network subnet
            with its corresponding UID limit.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("MaxAllowedUids", extract_value=extract_value)[
            "MaxAllowedUids"
        ]

    def query_map_min_stake(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of minimum allowed stake on the network.

        Queries the network to obtain the minimum number of stake,
        which is represented in nanotokens.

        Returns:
            A dictionary mapping network UIDs to
            their minimum allowed stake values.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("MinStake", extract_value=extract_value)["MinStake"]

    def query_map_max_stake(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of the maximum stake values for the network.

        Queries the network for the maximum stake values across various s
        ubnets of the network.

        Returns:
            A dictionary mapping network UIDs to their maximum stake values.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("MaxStake", extract_value=extract_value)["MaxStake"]

    def query_map_founder(self, extract_value: bool = False) -> dict[int, str]:
        """
        Retrieves a mapping of founders for the network.

        Queries the network to obtain the founders associated with
        various subnets.

        Returns:
            A dictionary mapping network UIDs to their respective founders.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("Founder", extract_value=extract_value)["Founder"]

    def query_map_founder_share(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of founder shares for the network.

        Queries the network for the share percentages
        allocated to founders across different subnets.

        Returns:
            A dictionary mapping network UIDs to their founder share percentages.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("FounderShare", extract_value=extract_value)[
            "FounderShare"
        ]

    def query_map_incentive_ratio(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of incentive ratios for the network.

        Queries the network for the incentive ratios,
        which are the proportions of rewards or incentives
        allocated in different subnets of the network.

        Returns:
            A dictionary mapping network UIDs to their incentive ratios.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("IncentiveRatio", extract_value=extract_value)[
            "IncentiveRatio"
        ]

    def query_map_trust_ratio(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of trust ratios for the network.

        Queries the network for trust ratios,
        indicative of the level of trust or credibility assigned
        to different subnets of the network.

        Returns:
            A dictionary mapping network UIDs to their trust ratios.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("TrustRatio", extract_value=extract_value)["TrustRatio"]

    def query_map_vote_mode_subnet(self, extract_value: bool = False) -> dict[int, str]:
        """
        Retrieves a mapping of vote modes for subnets within the network.

        Queries the network for the voting modes used in different
        subnets, which define the methodology or approach of voting within those
        subnets.

        Returns:
            A dictionary mapping network UIDs to their vote
            modes for subnets.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("VoteModeSubnet", extract_value=extract_value)[
            "VoteModeSubnet"
        ]

    def query_map_legit_whitelist(
        self, extract_value: bool = False
    ) -> dict[Ss58Address, int]:
        """
        Retrieves a mapping of whitelisted addresses for the network.

        Queries the network for a mapping of whitelisted addresses
        and their respective legitimacy status.

        Returns:
            A dictionary mapping addresses to their legitimacy status.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("LegitWhitelist", extract_value=extract_value)[
            "LegitWhitelist"
        ]

    def query_map_subnet_names(self, extract_value: bool = False) -> dict[int, str]:
        """
        Retrieves a mapping of subnet names within the network.

        Queries the network for the names of various subnets,
        providing an overview of the different
        subnets within the network.

        Returns:
            A dictionary mapping network UIDs to their subnet names.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("SubnetNames", extract_value=extract_value)["SubnetNames"]

    def query_map_balances(
        self, extract_value: bool = False
    ) -> dict[str, dict["str", int | dict[str, int]]]:
        """
        Retrieves a mapping of account balances within the network.

        Queries the network for the balances associated with different accounts.
        It provides detailed information including various types of
        balances for each account.

        Returns:
            A dictionary mapping account addresses to their balance details.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("Account", module="System", extract_value=extract_value)[
            "Account"
        ]

    def query_map_registration_blocks(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[int, int]:
        """
        Retrieves a mapping of registration blocks for UIDs on the network.

        Queries the network to find the block numbers at which various
        UIDs were registered.

        Args:
            netuid: The network UID from which to get the registrations.

        Returns:
            A dictionary mapping UIDs to their registration block numbers.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map(
            "RegistrationBlock", [netuid], extract_value=extract_value
        )["RegistrationBlock"]

    def query_map_name(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[int, str]:
        """
        Retrieves a mapping of names for keys on the network.

        Queries the network for the names associated with different keys.
        It provides a mapping of key UIDs to their registered names.

        Args:
            netuid: The network UID from which to get the names.

        Returns:
            A dictionary mapping key UIDs to their names.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("Name", [netuid], extract_value=extract_value)["Name"]

    #  == QUERY FUNCTIONS == #

    def get_immunity_period(self, netuid: int = 0) -> int:
        """
        Queries the network for the immunity period setting.

        The immunity period is a time duration during which a module
        can not be deregistered from the network.
        Fetches the immunity period for a specified network subnet.

        Args:
            netuid: The network UID for which to query the immunity period.

        Returns:
            The immunity period setting for the specified network subnet.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "ImmunityPeriod",
            params=[netuid],
        )

    def get_max_set_weights_per_epoch(self):
        return self.query("MaximumSetWeightCallsPerEpoch")

    def get_min_allowed_weights(self, netuid: int = 0) -> int:
        """
        Queries the network for the minimum allowed weights setting.

        Retrieves the minimum weight values that are possible to set
        by a validator within a specific network subnet.

        Args:
            netuid: The network UID for which to query the minimum allowed
              weights.

        Returns:
            The minimum allowed weight values for the specified network
              subnet.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "MinAllowedWeights",
            params=[netuid],
        )

    def get_max_allowed_weights(self, netuid: int = 0) -> int:
        """
        Queries the network for the maximum allowed weights setting.

        Retrieves the maximum weight values that are possible to set
        by a validator within a specific network subnet.

        Args:
            netuid: The network UID for which to query the maximum allowed
              weights.

        Returns:
            The maximum allowed weight values for the specified network
              subnet.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query("MaxAllowedWeights", params=[netuid])

    def get_max_allowed_uids(self, netuid: int = 0) -> int:
        """
        Queries the network for the maximum allowed UIDs setting.

        Fetches the upper limit on the number of user IDs that can
        be allocated or used within a specific network subnet.

        Args:
            netuid: The network UID for which to query the maximum allowed UIDs.

        Returns:
            The maximum number of allowed UIDs for the specified network subnet.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query("MaxAllowedUids", params=[netuid])

    def get_name(self, netuid: int = 0) -> str:
        """
        Queries the network for the name of a specific subnet.

        Args:
            netuid: The network UID for which to query the name.

        Returns:
            The name of the specified network subnet.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query("Name", params=[netuid])

    def get_subnet_name(self, netuid: int = 0) -> str:
        """
        Queries the network for the name of a specific subnet.

        Args:
            netuid: The network UID for which to query the name.

        Returns:
            The name of the specified network subnet.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query("SubnetNames", params=[netuid])

    def get_global_dao_treasury(self):
        return self.query("GlobalDaoTreasury")

    def get_n(self, netuid: int = 0) -> int:
        """
        Queries the network for the 'N' hyperparameter, which represents how
        many modules are on the network.

        Args:
            netuid: The network UID for which to query the 'N' hyperparameter.

        Returns:
            The value of the 'N' hyperparameter for the specified network
              subnet.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query("N", params=[netuid])

    def get_tempo(self, netuid: int = 0) -> int:
        """
        Queries the network for the tempo setting, measured in blocks, for the
        specified subnet.

        Args:
            netuid: The network UID for which to query the tempo.

        Returns:
            The tempo setting for the specified subnet.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query("Tempo", params=[netuid])

    def get_total_stake(self, netuid: int = 0):
        """
        Queries the network for the total stake amount.

        Retrieves the total amount of stake within a specific network subnet.

        Args:
            netuid: The network UID for which to query the total stake.

        Returns:
            The total stake amount for the specified network subnet.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "TotalStake",
            params=[netuid],
        )

    def get_registrations_per_block(self):
        """
        Queries the network for the number of registrations per block.

        Fetches the number of registrations that are processed per
        block within the network.

        Returns:
            The number of registrations processed per block.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "RegistrationsPerBlock",
        )

    def max_registrations_per_block(self, netuid: int = 0):
        """
        Queries the network for the maximum number of registrations per block.

        Retrieves the upper limit of registrations that can be processed in
        each block within a specific network subnet.

        Args:
            netuid: The network UID for which to query.

        Returns:
            The maximum number of registrations per block for
            the specified network subnet.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "MaxRegistrationsPerBlock",
            params=[netuid],
        )

    def get_proposal(self, proposal_id: int = 0):
        """
        Queries the network for a specific proposal.

        Args:
            proposal_id: The ID of the proposal to query.

        Returns:
            The details of the specified proposal.

        Raises:
            QueryError: If the query to the network fails, is invalid,
                or if the proposal ID does not exist.
        """

        return self.query(
            "Proposals",
            params=[proposal_id],
        )

    def get_trust(self, netuid: int = 0):
        """
        Queries the network for the trust setting of a specific network subnet.

        Retrieves the trust level or score, which may represent the
        level of trustworthiness or reliability within a
        particular network subnet.

        Args:
            netuid: The network UID for which to query the trust setting.

        Returns:
            The trust level or score for the specified network subnet.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "Trust",
            params=[netuid],
        )

    def get_uids(self, key: Ss58Address, netuid: int = 0) -> bool | None:
        """
        Queries the network for module UIDs associated with a specific key.

        Args:
            key: The key address for which to query UIDs.
            netuid: The network UID within which to search for the key.

        Returns:
            A list of UIDs associated with the specified key.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "Uids",
            params=[netuid, key],
        )

    def get_unit_emission(self) -> int:
        """
        Queries the network for the unit emission setting.

        Retrieves the unit emission value, which represents the
        emission rate or quantity for the $COMAI token.

        Returns:
            The unit emission value in nanos for the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query("UnitEmission")

    def get_tx_rate_limit(self) -> int:
        """
        Queries the network for the transaction rate limit.

        Retrieves the rate limit for transactions within the network,
        which defines the maximum number of transactions that can be
        processed within a certain timeframe.

        Returns:
            The transaction rate limit for the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "TxRateLimit",
        )

    def get_burn_rate(self) -> int:
        """
        Queries the network for the burn rate setting.

        Retrieves the burn rate, which represents the rate at
        which the $COMAI token is permanently
        removed or 'burned' from circulation.

        Returns:
            The burn rate for the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "BurnRate",
            params=[],
        )

    def get_burn(self, netuid: int = 0) -> int:
        """
        Queries the network for the burn setting.

        Retrieves the burn value, which represents the amount of the
        $COMAI token that is 'burned' or permanently removed from
        circulation.

        Args:
            netuid: The network UID for which to query the burn value.

        Returns:
            The burn value for the specified network subnet.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query("Burn", params=[netuid])

    def get_min_burn(self) -> int:
        """
        Queries the network for the minimum burn setting.

        Retrieves the minimum burn value, indicating the lowest
        amount of the $COMAI tokens that can be 'burned' or
        permanently removed from circulation.

        Returns:
            The minimum burn value for the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "MinBurn",
            params=[],
        )

    def get_min_weight_stake(self) -> int:
        """
        Queries the network for the minimum weight stake setting.

        Retrieves the minimum weight stake, which represents the lowest
        stake weight that is allowed for certain operations or
        transactions within the network.

        Returns:
            The minimum weight stake for the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query("MinWeightStake", params=[])

    def get_vote_mode_global(self) -> str:
        """
        Queries the network for the global vote mode setting.

        Retrieves the global vote mode, which defines the overall voting
        methodology or approach used across the network in default.

        Returns:
            The global vote mode setting for the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "VoteModeGlobal",
        )

    def get_max_proposals(self) -> int:
        """
        Queries the network for the maximum number of proposals allowed.

        Retrieves the upper limit on the number of proposals that can be
        active or considered at any given time within the network.

        Returns:
            The maximum number of proposals allowed on the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "MaxProposals",
        )

    def get_max_registrations_per_block(self) -> int:
        """
        Queries the network for the maximum number of registrations per block.

        Retrieves the maximum number of registrations that can
        be processed in each block within the network.

        Returns:
            The maximum number of registrations per block on the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "MaxRegistrationsPerBlock",
            params=[],
        )

    def get_max_name_length(self) -> int:
        """
        Queries the network for the maximum length allowed for names.

        Retrieves the maximum character length permitted for names
        within the network. Such as the module names

        Returns:
            The maximum length allowed for names on the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "MaxNameLength",
            params=[],
        )

    def get_global_vote_threshold(self) -> int:
        """
        Queries the network for the global vote threshold.

        Retrieves the global vote threshold, which is the critical value or
        percentage required for decisions in the network's governance process.

        Returns:
            The global vote threshold for the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "GlobalVoteThreshold",
        )

    def get_max_allowed_subnets(self) -> int:
        """
        Queries the network for the maximum number of allowed subnets.

        Retrieves the upper limit on the number of subnets that can
        be created or operated within the network.

        Returns:
            The maximum number of allowed subnets on the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "MaxAllowedSubnets",
            params=[],
        )

    def get_max_allowed_modules(self) -> int:
        """
        Queries the network for the maximum number of allowed modules.

        Retrieves the upper limit on the number of modules that
        can be registered within the network.

        Returns:
            The maximum number of allowed modules on the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "MaxAllowedModules",
            params=[],
        )

    def get_min_stake(self, netuid: int = 0) -> int:
        """
        Queries the network for the minimum stake required to register a key.

        Retrieves the minimum amount of stake necessary for
        registering a key within a specific network subnet.

        Args:
            netuid: The network UID for which to query the minimum stake.

        Returns:
            The minimum stake required for key registration in nanos.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query("MinStake", params=[netuid])

    def get_stake(
        self,
        key: Ss58Address,
        netuid: int = 0,
    ) -> int:
        """
        Queries the network for the stake delegated with a specific key.

        Retrieves the amount of total staked tokens
        delegated a specific key address

        Args:
            key: The address of the key to query the stake for.
            netuid: The network UID from which to get the query.

        Returns:
            The amount of stake held by the specified key in nanos.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "Stake",
            params=[netuid, key],
        )

    def get_stakefrom(
        self,
        key_addr: Ss58Address,
        netuid: int = 0,
    ) -> dict[str, int]:
        """
        Retrieves a list of keys from which a specific key address is staked.

        Queries the network for all the stakes received by a
        particular key from different sources.

        Args:
            key_addr: The address of the key to query stakes from.

            netuid: The network UID from which to get the query.

        Returns:
            A dictionary mapping key addresses to the amount of stake
            received from each.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """
        result = self.query("StakeFrom", [netuid, key_addr])

        return {k: v for k, v in result}

    def get_staketo(
        self,
        key_addr: Ss58Address,
        netuid: int = 0,
    ) -> dict[str, int]:
        """
        Retrieves a list of keys to which a specific key address stakes to.

        Queries the network for all the stakes made by a particular key to
        different destinations.

        Args:
            key_addr: The address of the key to query stakes to.

            netuid: The network UID from which to get the query.

        Returns:
            A dictionary mapping key addresses to the
            amount of stake given to each.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        result = self.query("StakeTo", [netuid, key_addr])

        return {k: v for k, v in result}

    def get_balance(
        self,
        addr: Ss58Address,
    ) -> int:
        """
        Retrieves the balance of a specific key.

        Args:
            addr: The address of the key to query the balance for.

        Returns:
            The balance of the specified key.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        result = self.query("Account", module="System", params=[addr])

        return result["data"]["free"]

    def get_block(self, block_hash: str | None = None) -> dict[Any, Any] | None:
        """
        Retrieves information about a specific block in the network.

        Queries the network for details about a block, such as its number,
        hash, and other relevant information.

        Returns:
            The requested information about the block,
            or None if the block does not exist
            or the information is not available.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        with self.get_conn() as substrate:
            block: dict[Any, Any] | None = substrate.get_block(  # type: ignore
                block_hash  # type: ignore
            )

        return block

    def get_existential_deposit(self, block_hash: str | None = None) -> int:
        """
        Retrieves the existential deposit value for the network.

        The existential deposit is the minimum balance that must be maintained
        in an account to prevent it from being purged. Denotated in nano units.

        Returns:
            The existential deposit value in nano units.
        Note:
            The value returned is a fixed value defined in the
            client and may not reflect changes in the network's configuration.
        """

        with self.get_conn() as substrate:
            result: int = substrate.get_constant(  #  type: ignore
                "Balances", "ExistentialDeposit", block_hash
            ).value  #  type: ignore

        return result
