import networkx as nx


def read_hog(file: str) -> tuple[nx.Graph, dict]:
    with open(file, "r") as file:
        pos = dict()
        graph = nx.Graph()

        _ = file.readline()  # read number of vertices

        for i, line in enumerate(file.readlines()):
            x, y, *neighbours = line.strip().split(" ")
            pos[i] = (float(x), float(y))

            for u in neighbours:
                graph.add_edge(int(u), i)

    return graph, pos


if __name__ == '__main__':
    g, p = read_hog("../house_of_graphs/1004.txt")
    print(g, p)