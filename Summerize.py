import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from enum import Enum
from util import PCA, SUBSPACE

pcaTool = PCA.WeightPCA()
smTool = SUBSPACE.SubspaceDiff()

class DistType(Enum):
    LINEAR = 1
    RBF = 2
    GRASSMANN = 3

class ListSum():
    def __init__(self):
        pass

    def print_mean(self, container):
        print("mean : {:.4f}".format(np.mean(container)))

    def print_var(self, container):
        print("mean : {:.4f}".format(np.var(container)))

    def print_std(self, container):
        print("mean : {:.4f}".format(np.std(container)))

class PluralProjSum():
    def __init__(self, trajNum):
        self.trajNum = trajNum

    def sum_intrinsic_dim(self, projections, period=10):
        print("="*20, end="")
        print("PluralProjSum - sum_intrinsic_dim", end="")
        print("="*20)
        sampleNum = projections.shape[1]
        sampleNum = int(sampleNum / self.trajNum)
        for trajIdx in range(self.trajNum):
            start = trajIdx*sampleNum
            end = (trajIdx+1)*sampleNum
            X = projections[:,start:end]
            _, _ = pcaTool.pca_basic(X, isVerbose=True, period=period)
        print("="*50)
        print()

    def sum_traj_sim(self, projections):
        print("="*20, end="")
        print("PluralProjSum - sum_traj_sim", end="")
        print("="*20)
        sampleNum = projections.shape[1]
        sampleNum = int(sampleNum / self.trajNum)
        for trajIdx_i in range(self.trajNum):
            start_i = trajIdx_i*sampleNum
            end_i = (trajIdx_i+1)*sampleNum
            i_vec = projections[:,end_i-1] - projections[:,start_i]
            i_vec = i_vec / np.linalg.norm(i_vec)
            for trajIdx_j in range(self.trajNum):
                start_j = trajIdx_j*sampleNum
                end_j = (trajIdx_j+1)*sampleNum
                j_vec = projections[:,end_j-1] - projections[:,start_j]
                j_vec = j_vec / np.linalg.norm(j_vec)
                print("{:.4f}".format(cp.dot(i_vec, j_vec)), end=" || ")
            print()
        print("="*50)
        print()

    def sum_distance(self, projections):
        print("="*20, end="")
        print("PluralProjSum - sum_distance", end="")
        print("="*20)
        sampleNum = projections.shape[1]
        sampleNum = int(sampleNum / self.trajNum)

        print("1. start-end")
        for trajIdx in range(self.trajNum):
            start = trajIdx*sampleNum
            end = (trajIdx+1)*sampleNum
            start_vec = projections[:,start]
            end_vec = projections[:,end-1]
            print("{:.4f}".format(np.linalg.norm(start_vec - end_vec)), end=" || ")
        print()

        print("2. origin-start")
        for trajIdx in range(self.trajNum):
            start = trajIdx*sampleNum
            start_vec = projections[:,start]
            print("{:.4f}".format(np.linalg.norm(start_vec)), end=" || ")
        print()

        print("3. origin-end")
        for trajIdx in range(self.trajNum):
            end = (trajIdx+1)*sampleNum
            end_vec = projections[:,end-1]
            print("{:.4f}".format(np.linalg.norm(end_vec)), end=" || ")
        print()
        
        print("4. between start")
        for trajIdx_i in range(self.trajNum):
            start_i = trajIdx_i*sampleNum
            start_vec_i = projections[:,start_i]
            for trajIdx_j in range(self.trajNum):
                start_j = trajIdx_j*sampleNum
                start_vec_j = projections[:,start_j]
                print("{:.4f}".format(np.linalg.norm(start_vec_i - start_vec_j)), end=" || ")
            print()
        print()

        print("5. between end")
        for trajIdx_i in range(self.trajNum):
            end_i = (trajIdx_i+1)*sampleNum - 1
            end_vec_i = projections[:,end_i]
            for trajIdx_j in range(self.trajNum):
                end_j = (trajIdx_j+1)*sampleNum - 1
                end_vec_j = projections[:,end_j]
                print("{:.4f}".format(np.linalg.norm(end_vec_i - end_vec_j)), end=" || ")
            print()
        print()
        print("="*50)
        print()

    def sum_cca(self, projections):
        print("="*20, end="")
        print("PluralProjSum - sum_cca", end="")
        print("="*20)
        sampleNum = projections.shape[1]
        sampleNum = int(sampleNum / self.trajNum)
        basisContainer = []
        for trajIdx in range(self.trajNum):
            start = trajIdx*sampleNum
            end = (trajIdx+1)*sampleNum
            X = projections[:,start:end]
            _, basis = pcaTool.pca_basic(X)
            basis = basis[:,:sampleNum]
            basisContainer.append(basis)

        magContainer = []
        magMatrix = cp.zeros((self.trajNum,self.trajNum))
        for trajIdx_i in range(self.trajNum):
            for trajIdx_j in range(trajIdx_i+1,self.trajNum):
                print("cosine sim list between subspace {:02d} and {:02d}".format(trajIdx_i, trajIdx_j))
                tmp = smTool.calc_magnitude(basisContainer[trajIdx_i], basisContainer[trajIdx_j], isVerbose=True)
                magContainer.append(tmp.get())
                magMatrix[trajIdx_i,trajIdx_j] = tmp
                magMatrix[trajIdx_j,trajIdx_i] = tmp
        
        for i in range(self.trajNum):
            for j in range(self.trajNum):
                print("{:.4f}".format(magMatrix[i,j]), end=" || ")
            print()

        print("cca mean : {:.4f}".format(np.mean(magContainer)))
        print("cca var : {:.4f}".format(np.var(magContainer)))
        print("="*50)
        print()


# result container
# ㄴ container1 : m1_layer1, m2_layer1, ... m100_layer1
# ㄴ container2 : m1_layer2, m2_layer2, ... m100_layer2
# ...
class SingleNestedListPlot():
    def __init__(self, model_name):
        self.model_name = model_name

    def plot_first_magnitude(self, result, label, seed, num):
        plt.title("1st magnitude of weight subspace")
        plt.xlabel("each epoch")
        plt.ylabel("1st mangitude")
        for lb, rs in zip(label, result):
            plt.plot(range(len(rs)), rs, label=lb)
        plt.grid()
        plt.legend()
        plt.savefig("result/firstmag_{}_{:04d}_{:02d}.png".format(self.model_name, seed, num))
        plt.clf()

    def plot_second_magnitude(self, result, label, seed, num):
        plt.title("2nd magnitude of weight subspace")
        plt.xlabel("each epoch")
        plt.ylabel("2nd mangitude")
        for lb, rs in zip(label, result):
            plt.plot(range(len(rs)), rs, label=lb)
        plt.grid()
        plt.legend()
        plt.savefig("result/secondmag_{}_{:04d}_{:02d}.png".format(self.model_name, seed, num))
        plt.clf()

# result container
# ㄴ layer container1
#   ㄴ container_along : malong1_layer1, malong2_layer1, ... malong100_layer1
#   ㄴ container_orth : morth1_layer1, morth2_layer1, ... morth100_layer1
# ㄴ layer container2
#   ㄴ container_along : malong1_layer2, malong2_layer2, ... malong100_layer2
#   ㄴ container_orth : morth1_layer2, morth2_layer2, ... morth100_layer2
# ...
class DoubleNestedListPlot():
    def __init__(self, model_name):
        self.model_name = model_name

    def plot_first_decomposition(self, result, label, seed, num):
        plt.title("1st magnitude of weight subspace(along geodesic)")
        plt.xlabel("each epoch")
        plt.ylabel("1st magnitude")
        for lb, rs in zip(label, result):
            plt.plot(range(len(rs[0])), rs[0], label=lb)
        plt.grid()
        plt.legend()
        plt.savefig("result/firstmag_along_{}_{:04d}_{:02d}.png".format(self.model_name, seed, num))
        plt.clf()

        plt.title("1st magnitude of weight subspace(orthogonal to geodesic)")
        plt.xlabel("each epoch")
        plt.ylabel("1st magnitude")
        for lb, rs in zip(label, result):
            plt.plot(range(len(rs[1])), rs[1], label=lb)
        plt.grid()
        plt.legend()
        plt.savefig("result/firstmag_orth_{}_{:04d}_{:02d}.png".format(self.model_name, seed, num))
        plt.clf()

        plt.title("1st magnitude of weight subspace(sum)")
        plt.xlabel("each epoch")
        plt.ylabel("1st magnitude")
        for lb, rs in zip(label, result):
            plt.plot(range(len(rs[0]+rs[1])), rs[0]+rs[1], label=lb)
        plt.grid()
        plt.legend()
        plt.savefig("result/firstmag_sum_{}_{:04d}_{:02d}.png".format(self.model_name, seed, num))
        plt.clf()

    def plot_second_decomposition(self, result, label, seed, num):
        plt.title("2nd magnitude of weight subspace(along geodesic)")
        plt.xlabel("each epoch")
        plt.ylabel("2nd magnitude")
        for lb, rs in zip(label, result):
            plt.plot(range(len(rs[0])), rs[0], label=lb)
        plt.grid()
        plt.legend()
        plt.savefig("result/secondmag_along_{}_{:04d}_{:02d}.png".format(self.model_name, seed, num))
        plt.clf()

        plt.title("2nd magnitude of weight subspace(orthogonal to geodesic)")
        plt.xlabel("each epoch")
        plt.ylabel("2nd magnitude")
        for lb, rs in zip(label, result):
            plt.plot(range(len(rs[1])), rs[1], label=lb)
        plt.grid()
        plt.legend()
        plt.savefig("result/secondmag_orth_{}_{:04d}_{:02d}.png".format(self.model_name, seed, num))
        plt.clf()

        plt.title("2nd magnitude of weight subspace(sum)")
        plt.xlabel("each epoch")
        plt.ylabel("2nd magnitude")
        for lb, rs in zip(label, result):
            plt.plot(range(len(rs[0]+rs[1])), rs[0]+rs[1], label=lb)
        plt.grid()
        plt.legend()
        plt.savefig("result/secondmag_sum_{}_{:04d}_{:02d}.png".format(self.model_name, seed, num))
        plt.clf()

    def plot_first_decomposition_layerwise(self, result, label, seed, num):
        for lb, rs in zip(label, result):
            plt.title("1st magnitude of weight subspace({})".format(lb))
            plt.xlabel("each epoch")
            plt.ylabel("1st magnitude")

            plt.plot(range(len(rs[0]+rs[1])), rs[0]+rs[1], color="green", label="sum")
            plt.plot(range(len(rs[0])), rs[0], ':', color="red", label="along to geodesic")
            plt.plot(range(len(rs[1])), rs[1], ':', color="blue", label="orthogonal to geodesic")
            plt.grid()
            plt.legend()
            plt.savefig("result/firstmag_decomposed_{}_{:04d}_{:02d}_{}.png".format(self.model_name, seed, num, lb))
            plt.clf()

    def plot_second_decomposition_layerwise(self, result, label, seed, num):
        for lb, rs in zip(label, result):
            plt.title("2nd magnitude of weight subspace({})".format(lb))
            plt.xlabel("each epoch")
            plt.ylabel("2nd magnitude")

            plt.plot(range(len(rs[0]+rs[1])), rs[0]+rs[1], color="green", label="sum")
            plt.plot(range(len(rs[0])), rs[0], ':', color="red", label="along to geodesic")
            plt.plot(range(len(rs[1])), rs[1], ':', color="blue", label="orthogonal to geodesic")
            plt.grid()
            plt.legend()
            plt.savefig("result/secondmag_decomposed_{}_{:04d}_{:02d}_{}.png".format(self.model_name, seed, num, lb))
            plt.clf()

    def plot_geodesic_decomposition(self, result, label, seed, num):
        plt.title("geodesic decomposition of weight subspace(along geodesic)")
        plt.xlabel("each epoch")
        plt.ylabel("along component")
        for lb, rs in zip(label, result):
            plt.plot(range(len(rs[0])), rs[0], label=lb)
        plt.grid()
        plt.legend()
        plt.savefig("result/gd_along_{}_{:04d}_{:02d}.png".format(self.model_name, seed, num))
        plt.clf()

        plt.title("geodesic decomposition of weight subspace(orthogonal to geodesic)")
        plt.xlabel("each epoch")
        plt.ylabel("orth component")
        for lb, rs in zip(label, result):
            plt.plot(range(len(rs[1])), rs[1], label=lb)
        plt.grid()
        plt.legend()
        plt.savefig("result/gd_orth_{}_{:04d}_{:02d}.png".format(self.model_name, seed, num))
        plt.clf()

        plt.title("geodesic decomposition of weight subspace(sum)")
        plt.xlabel("each epoch")
        plt.ylabel("along + orth")
        for lb, rs in zip(label, result):
            plt.plot(range(len(rs[0]+rs[1])), np.sqrt(np.power(rs[0], 2)+np.power(rs[1], 2)), label=lb)
        plt.grid()
        plt.legend()
        plt.savefig("result/gd_sum_{}_{:04d}_{:02d}.png".format(self.model_name, seed, num))
        plt.clf()

    def plot_geodesic_decomposition_layerwise(self, result, label, seed, num):
        for lb, rs in zip(label, result):
            plt.title("geodesic decomposition of weight subspace({})".format(lb))
            plt.xlabel("each epoch")
            plt.ylabel("each component")

            plt.plot(range(len(rs[0]+rs[1])), np.sqrt(np.power(rs[0], 2)+np.power(rs[1], 2)), color="green", label="sum")
            plt.plot(range(len(rs[0])), rs[0], ':', color="red", label="along to geodesic")
            plt.plot(range(len(rs[1])), rs[1], ':', color="blue", label="orthogonal to geodesic")
            plt.grid()
            plt.legend()
            plt.savefig("result/gd_decomposed_{}_{:04d}_{:02d}_{}.png".format(self.model_name, seed, num, lb))
            plt.clf()

# result container should be double-nested!
class ProjPlot():
    def __init__(self, model_name):
        self.model_name = model_name
        self.title_font = {
            'fontsize' : 10,
            'fontweight' : 'bold'
        }
        self.sub_font = {
            'fontsize' : 7
        }
        self.sub_font_3d = {
            'fontsize' : 5
        }
        self.colorList = ["blue", "red", "green", "yellow" "black", 
            "orange", "purple", "pink", "grey","skyblue",
            "olive", "brown", "cyan", "navy", "lime",
             "gold", "crimson", "indigo", "darkgreen", "ivory"]
        self.cmapList = ["spring", "summer", "autumn", "winter", "cool", "hot", "copper", "plasma", "inferno", "magma"]

    def plot_2d_singular(self, result, label, seed, num, gradientColor=True, dist="LINEAR"):
        distance = DistType[dist]
        iterNum = len(result)
        for i in range(iterNum):
            plt.title("pca projection of weight subspace", loc='left', fontdict=self.title_font)
            plt.title("{}".format(label[i]), loc='right', fontdict=self.sub_font, pad=30)
            plt.title("Distance : {}".format(distance.name), loc='right', fontdict=self.sub_font, pad=10)
            plt.grid()
            if gradientColor:
                sampleNum = len(result[i])
                t = np.arange(1, sampleNum+1)
                plt.scatter(result[i][0], result[i][1], c=t, cmap=self.cmapList[i])
                plt.colorbar()
            else:
                plt.scatter(result[i][0], result[i][1], color=self.colorList[i])
            plt.savefig("result/PCA2D_{}_{}_{:04d}_{:02d}_{}.png".format(self.model_name, distance.name, seed, num, label[i]))
            plt.clf()

    def plot_3d_singular(self, result, label, seed, num, gradientColor=True, dist="LINEAR"):
        distance = DistType[dist]
        iterNum = len(result)
        for i in range(iterNum):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("pca projection of weight subspace", loc='left', fontdict=self.title_font, pad=500)
            ax.set_title("{}".format(label[i]), loc='right', fontdict=self.sub_font_3d, pad=30)
            ax.set_title("Distance : {}".format(distance.name), loc='right', fontdict=self.sub_font_3d, pad=10)
            if gradientColor:
                sampleNum = len(result[i])
                t = np.arange(1, sampleNum+1)
                p = ax.scatter(result[i][0], result[i][1], result[i][2], c=t, cmap=self.cmapList[i])
                fig.colorbar(p, ax=ax)
            else:
                ax.scatter(result[i][0], result[i][1], result[i][2], color=self.colorList[i])
            plt.savefig("result/PCA3D_{}_{}_{:04d}_{:02d}_{}.png".format(self.model_name, distance.name, seed, num, label[i]))
            plt.clf()
    
    def plot_2d_plural(self, result, label, trajNum, seedContainer, num, gradientColor=True, dist="LINEAR"):
        distance = DistType[dist]
        iterNum = len(result)
        sc = list(map(str, seedContainer))
        seedStr = "_".join(sc)
        for i in range(iterNum):
            plt.title("pca projection of weight subspace", loc='left', fontdict=self.title_font)
            plt.title("{}".format(label[i]), loc='right', fontdict=self.sub_font, pad=30)
            plt.title("Distance : {}".format(distance.name), loc='right', fontdict=self.sub_font, pad=10)
            plt.grid()
            if gradientColor:
                sampleNum = len(result[i][0])
                sampleNum = int(sampleNum / trajNum)
                t = np.arange(1, sampleNum+1)
                for sampleIdx in range(trajNum):
                    start = sampleIdx*sampleNum
                    end = (sampleIdx+1)*sampleNum
                    plt.scatter(result[i][0][start:end], result[i][1][start:end], c=t, cmap=self.cmapList[sampleIdx%10], label="traj(layer) {:02d}".format(sampleIdx+1))
            else:
                sampleNum = len(result[i][0])
                sampleNum = int(sampleNum / trajNum)
                for sampleIdx in range(trajNum):
                    start = sampleIdx*sampleNum
                    end = (sampleIdx+1)*sampleNum
                    plt.scatter(result[i][0][start:end], result[i][1][start:end], color=self.colorList[sampleIdx%20], label="traj(layer) {:02d}".format(sampleIdx+1))
            plt.legend()
            plt.savefig("result/PCA2D_{}_{}_{}_{:02d}_{}.png".format(self.model_name, distance.name, seedStr, num, label[i]))
            plt.clf()

    def plot_3d_plural(self, result, label, trajNum, seedContainer, num, gradientColor=True, dist="LINEAR"):
        distance = DistType[dist]
        iterNum = len(result)
        sc = list(map(str, seedContainer))
        seedStr = "_".join(sc)
        for i in range(iterNum):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("pca projection of weight subspace", loc='left', fontdict=self.title_font, pad=500)
            ax.set_title("{}".format(label[i]), loc='right', fontdict=self.sub_font_3d, pad=30)
            ax.set_title("Distance : {}".format(distance.name), loc='right', fontdict=self.sub_font_3d, pad=10)
            if gradientColor:
                sampleNum = len(result[i][0])
                sampleNum = int(sampleNum / trajNum)
                t = np.arange(1, sampleNum+1)
                for sampleIdx in range(trajNum):
                    start = sampleIdx*sampleNum
                    end = (sampleIdx+1)*sampleNum
                    ax.scatter(result[i][0][start:end], result[i][1][start:end], result[i][2][start:end], c=t, cmap=self.cmapList[sampleIdx%10], label="traj(layer) {:02d}".format(sampleIdx+1))
            else:
                sampleNum = len(result[i][0])
                sampleNum = int(sampleNum / trajNum)
                for sampleIdx in range(trajNum):
                    start = sampleIdx*sampleNum
                    end = (sampleIdx+1)*sampleNum
                    ax.scatter(result[i][0][start:end], result[i][1][start:end], result[i][2][start:end], color=self.colorList[sampleIdx%20], label="traj(layer) {:02d}".format(sampleIdx+1))
            plt.legend()
            plt.savefig("result/PCA3D_{}_{}_{}_{:02d}_{}.png".format(self.model_name, distance.name, seedStr, num, label[i]))
            plt.clf()