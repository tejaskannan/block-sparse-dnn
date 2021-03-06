
import matplotlib.pyplot as plt

bsmm_dict = {}
tfbs_dict = {}
openai_dict = {}
dicts = {"bsmm" : bsmm_dict, "tfbs" : tfbs_dict, "openai" : openai_dict}
for str in ["bsmm", "tfbs", "openai"]:
    for matrixDim in [64, 256, 512]:
        for sparseDenom in [.1, .05, .02, .01]:
            for blockSize in [8, 16, 32]:
                with open(f"DATA{str}_bench.py{matrixDim}{blockSize}{sparseDenom}", 'r') as f:
                    line = f.readline()
                    if not line:
                        break
                    if matrixDim not in dicts[str]:
                        dicts[str][matrixDim] = {}
                    if sparseDenom not in dicts[str][matrixDim]:
                        dicts[str][matrixDim][sparseDenom] = {}
                    dicts[str][matrixDim][sparseDenom][blockSize] = 1000.0/float(line)

markers = {"bsmm" : '+', "tfbs" : '^', "openai" : 'o'}
colors = {.1 : 'b', .05 : 'g', .02: 'm', .01: 'k'}
for matrixDim in [64, 256, 512]:
    for strat in dicts.keys():
        for sparseDenom in dicts[strat][matrixDim].keys():
            lst = sorted(dicts[strat][matrixDim][sparseDenom].items())
            x, y = zip(*lst)

            plt.plot(x,y, f"{markers[strat]}-{colors[sparseDenom]}", label=f"{strat} {sparseDenom}-dense")
    plt.xlabel("Block dimension")
    plt.ylabel("Throughput (MatMul/s)")
    plt.legend()
    plt.title(f"Block Sparse MatMul Performance, {matrixDim}-dimensional")
    plt.savefig(f"bsmm_perf_{matrixDim}.png")
    plt.close()