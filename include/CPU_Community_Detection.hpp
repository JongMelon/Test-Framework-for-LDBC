#pragma once
#include<iostream>
#include<vector>
#include<fstream>
#include<chrono>
#include<numeric>
#include<algorithm>
#include<limits>
#include <graph_structure/graph_structure.hpp>
using namespace std;
static int CPU_CD_GRAPHSIZE;


static vector<int> cpu_cd_row_ptr, cpu_cd_col_indices; 
static vector<int> labels;

template <typename T>
void make_csr(graph_structure<T> &graph) {
    CPU_CD_GRAPHSIZE = graph.size();
    // cout<<CPU_CD_GRAPHSIZE<<endl;
    cpu_cd_row_ptr.resize(CPU_CD_GRAPHSIZE + 1);
    cpu_cd_row_ptr[0] = 0;
    CSR_graph<int> ARRAY_graph;
    ARRAY_graph=graph.toCSR();
    cpu_cd_row_ptr=ARRAY_graph.OUTs_Neighbor_start_pointers;
    cpu_cd_col_indices=ARRAY_graph.OUTs_Edges;
    // for (int i = 0; i < CPU_CD_GRAPHSIZE; i++) {
    //     for (int j :graph.ADJs[i]) {
    //         cpu_cd_col_indices.push_back(j.first);
    //     }
    //     cpu_cd_row_ptr[i + 1] = cpu_cd_row_ptr[i] + graph.ADJs[i].size();
    // }
}

void init_label() {
    for (int i = 0; i < CPU_CD_GRAPHSIZE; i++) {
        labels[i] = i;
    }
}

int findMostFrequentLabel(int start, int end) {
    vector<int> frequencyMap(CPU_CD_GRAPHSIZE, 0);
    int mostFre = -1, mostFreLab = -1;
    for (int i = start; i < end; i++) {
        int neighbor = cpu_cd_col_indices[i];
        frequencyMap[labels[neighbor]]++;
        if (frequencyMap[labels[neighbor]] > mostFre) {
            mostFre = frequencyMap[labels[neighbor]];
            mostFreLab = labels[neighbor];
        }
        else if (frequencyMap[labels[neighbor]] == mostFre && labels[neighbor] < mostFreLab) {
            mostFreLab = labels[neighbor];
        }
    }

    return mostFreLab;
}

void labelPropagation() {
    bool keepUpdating = true;
    while (keepUpdating) {
        keepUpdating = false;
        for (int i = 0; i < CPU_CD_GRAPHSIZE; ++i) {
            int start = cpu_cd_row_ptr[i], end = cpu_cd_row_ptr[i + 1];
            if (start == end) continue; 

            int mostFrequentLabel = findMostFrequentLabel(start, end);
            if (labels[i] != mostFrequentLabel) {
                labels[i] = mostFrequentLabel;
                keepUpdating = true;
            }
        }
    }
}

template <typename T>
int CPU_Community_Detection(graph_structure<T>& graph) {
    make_csr(graph, CPU_CD_GRAPHSIZE);
    init_label();
    labelPropagation();

    return 0;
}












