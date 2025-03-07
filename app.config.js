module.exports = {
  apps : [{
    name   : 'distributed_training_miner',
    script : 'neurons/miner.py',
    interpreter: 'python3',
    min_uptime: '5m',
    max_restarts: '5',
    args: ['--netuid','38','--subtensor.network','test','--wallet.name','test_dt','--wallet.hotkey','miner_1','--axon.port','22038','--dht.port','22039','--dht.ip','194.68.245.44','--axon.ip','194.68.245.44','--neuron.global_model_name','distributed/optimized-gpt2-1b-vtestnet-v1','--neuron.wandb_project','distributed_training_test','--neuron.initial_peers','/ip4/161.97.156.125/tcp/8001/p2p/12D3KooWC2c83iNT3b92Ngnjpe6ZLT1HQuyjBf8AiMx5JNhr5QzQ','--neuron.run_id','v0_4_6_r1_testnet','--neuron.local_model_name','kmfoda/gpt2-1b-miner-1-test','--neuron.min_group_size','1','--neuron.epoch_length','2']
  }]
}
