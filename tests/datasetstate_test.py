import hivemind
from bitarray import bitarray

from template.validator.validator_core import DatasetState

# Initialize the dummy data and the DatasetState instance
dataset_length = 96015  # Reduced size for testing
dataset_indices = bitarray("0" * dataset_length)  # Initialize all indices to 0 (unused)
# dataset_dict = {}  # Dummy DHT, aka dht_state
dht = hivemind.DHT(start=True)
run_id = "dummy_run_id"

dataset_state = DatasetState(dht, dataset_indices, run_id)

# Test get_dataset_indices
selected_groups = dataset_state.get_dataset_indices(2, 500)
print("Selected groups for training:", selected_groups)
print(dataset_state.dataset_indices_train.count(0))
selected_groups = dataset_state.get_dataset_indices(2, 500)
print("Selected groups for training:", selected_groups)
print(dataset_state.dataset_indices_train.count(0))

# Test get_dataset_indices_test with DHT
test_indices = dataset_state.get_dataset_indices_test(20)
print(test_indices)
print(dht.get("dataset_indices_test"))
dht.shutdown()
