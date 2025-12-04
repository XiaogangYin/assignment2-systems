for data_size in 1_000_000 10_000_000 100_000_000 1000_000_000
do
  for world_size in 2 4 6
  do
     uv run python cs336_systems/distributed_communication_single_node.py $world_size $data_size
  done
done

