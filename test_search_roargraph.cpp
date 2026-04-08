#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "../include/efanna2e/distance.h"
#include "../include/efanna2e/neighbor.h"
#include "../include/efanna2e/parameters.h"
#include "../include/efanna2e/util.h"
#include "../include/index_bipartite.h"

namespace po = boost::program_options;

float ComputeRecall(uint32_t q_num, uint32_t k, uint32_t gt_dim, uint32_t *res, uint32_t *gt) {
    uint32_t total_count = 0;
    for (uint32_t i = 0; i < q_num; i++) {
        std::vector<uint32_t> one_gt(gt + i * gt_dim, gt + i * gt_dim + k);
        std::vector<uint32_t> intersection;
        std::vector<uint32_t> temp_res(res + i * k, res + i * k + k);
        for (auto p : one_gt) {
            if (std::find(temp_res.begin(), temp_res.end(), p) != temp_res.end()) intersection.push_back(p);
        }

        total_count += static_cast<uint32_t>(intersection.size());
    }
    return static_cast<float>(total_count) / (float)(k * q_num);
}

static float ComputeRecallAtKPerQuery(const uint32_t *res, uint32_t res_k, const uint32_t *gt, uint32_t gt_dim,
                                      uint32_t k_eval) {
    if (k_eval == 0) {
        return 0.0f;
    }
    const uint32_t k_eff = std::min<uint32_t>({res_k, gt_dim, k_eval});
    std::vector<uint32_t> one_gt(gt, gt + k_eff);
    std::vector<uint32_t> temp_res(res, res + k_eff);
    uint32_t hit = 0;
    for (auto p : one_gt) {
        if (std::find(temp_res.begin(), temp_res.end(), p) != temp_res.end()) {
            hit++;
        }
    }
    return static_cast<float>(hit) / static_cast<float>(k_eff);
}

static std::vector<float> ComputeRecallPerQuery(uint32_t q_num, uint32_t k, uint32_t gt_dim, uint32_t *res,
                                                uint32_t *gt) {
    std::vector<float> recalls(q_num, 0.0f);
    for (uint32_t i = 0; i < q_num; i++) {
        const uint32_t *gt_begin = gt + i * gt_dim;
        const uint32_t *res_begin = res + i * k;

        std::vector<uint32_t> one_gt(gt_begin, gt_begin + k);
        std::vector<uint32_t> temp_res(res_begin, res_begin + k);
        uint32_t hit = 0;
        for (auto p : one_gt) {
            if (std::find(temp_res.begin(), temp_res.end(), p) != temp_res.end()) {
                hit++;
            }
        }
        recalls[i] = static_cast<float>(hit) / static_cast<float>(k);
    }
    return recalls;
}

static float QuantileNearestRank(std::vector<float> values, float q01) {
    if (values.empty()) {
        return 0.0f;
    }
    if (q01 <= 0.0f) {
        return *std::min_element(values.begin(), values.end());
    }
    if (q01 >= 1.0f) {
        return *std::max_element(values.begin(), values.end());
    }
    const size_t n = values.size();
    const size_t idx = static_cast<size_t>(std::ceil(q01 * static_cast<float>(n))) - 1;
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

double ComputeRderr(float* gt_dist, uint32_t gt_dim, std::vector<std::vector<float>>& res_dists,
                    uint32_t k, efanna2e::Metric metric) {
    double total_err = 0;
    uint32_t q_num = res_dists.size();

    for (uint32_t i = 0; i < q_num; i++) {
        std::vector<float> one_gt(gt_dist + i * gt_dim, gt_dist + i * gt_dim + k);
        std::vector<float> temp_res(res_dists[i].begin(), res_dists[i].end());
        if (metric == efanna2e::INNER_PRODUCT) {
            for (size_t j = 0; j < k; ++j) {
                temp_res[j] = -1.0 * temp_res[j];
            }
        } else if (metric == efanna2e::COSINE) {
            for (size_t j = 0; j < k; ++j) {
                temp_res[j] = 2.0 * (1.0 - (-1.0 * temp_res[j]));
            }
        }
        double err = 0.0;
        for (uint32_t j = 0; j < k; j++) {
            err += std::fabs(temp_res[j] - one_gt[j]) / double(one_gt[j]);
        }
        err = err / static_cast<double>(k);
        total_err = total_err + err;
    }
    return total_err / static_cast<double>(q_num);
}

int main(int argc, char **argv) {
    std::string base_data_file;
    std::string query_file;
    std::string sampled_query_data_file;
    std::string gt_file;

    std::string bipartite_index_save_file, projection_index_save_file;
    std::string data_type;
    std::string dist;
    std::vector<uint32_t> L_vec;
    uint32_t num_threads;
    uint32_t k;
    std::string evaluation_save_path = "";
    std::string per_query_csv_path = "";

    std::string search_mode = "greedy";
    uint32_t max_jump_candidates = 0;
    std::string per_query_log_path = "search_stats.log";

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist", po::value<std::string>(&dist)->required(), "distance function <l2/ip>");
        desc.add_options()("base_data_path", po::value<std::string>(&base_data_file)->required(),
                           "Input data file in bin format");
        desc.add_options()("query_path", po::value<std::string>(&query_file)->required(), "Query file in bin format");
        desc.add_options()("gt_path", po::value<std::string>(&gt_file)->required(), "Groundtruth file in bin format");
        desc.add_options()("projection_index_save_path",
                           po::value<std::string>(&projection_index_save_file)->required(),
                           "Path prefix for saving projetion index file components");
        desc.add_options()("L_pq", po::value<std::vector<uint32_t>>(&L_vec)->multitoken()->required(),
                           "Priority queue length for searching");
        desc.add_options()("k", po::value<uint32_t>(&k)->default_value(1)->required(), "k nearest neighbors");
        desc.add_options()("evaluation_save_path", po::value<std::string>(&evaluation_save_path),
                           "Path prefix for saving evaluation results");
        desc.add_options()("per_query_csv", po::value<std::string>(&per_query_csv_path),
                           "Path for saving per-query evaluation CSV");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads used for building index (defaults to omp_get_num_procs())");

        desc.add_options()("search_mode", po::value<std::string>(&search_mode)->default_value("greedy"),
                           "search mode <greedy/jump>");
        desc.add_options()("max_jump_candidates", po::value<uint32_t>(&max_jump_candidates)->default_value(0),
                           "max number of jump nn candidates to try enqueue; 0 means all");
        desc.add_options()("per_query_log", po::value<std::string>(&per_query_log_path)->default_value("search_stats.log"),
                           "log file path for per-query search statistics");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    if (search_mode != "greedy" && search_mode != "jump") {
        std::cerr << "search_mode must be either 'greedy' or 'jump'" << std::endl;
        return -1;
    }

    uint32_t base_num, base_dim, sq_num = 0, sq_dim = 0;
    efanna2e::load_meta<float>(base_data_file.c_str(), base_num, base_dim);
    if (!sampled_query_data_file.empty()) {
        efanna2e::load_meta<float>(sampled_query_data_file.c_str(), sq_num, sq_dim);
    }

    efanna2e::Parameters parameters;

    parameters.Set<uint32_t>("num_threads", num_threads);
    omp_set_num_threads(num_threads);

    uint32_t q_pts, q_dim;
    efanna2e::load_meta<float>(query_file.c_str(), q_pts, q_dim);
    float *query_data = nullptr;
    efanna2e::load_data<float>(query_file.c_str(), q_pts, q_dim, query_data);
    float *aligned_query_data = efanna2e::data_align(query_data, q_pts, q_dim);

    uint32_t gt_pts = 0, gt_dim = 0;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    {
        std::ifstream gt_in(gt_file, std::ios::binary);
        if (!gt_in.is_open()) {
            std::cerr << "open GT file error: " << gt_file << std::endl;
            return -1;
        }
        gt_in.read(reinterpret_cast<char*>(&gt_pts), 4);
        gt_in.read(reinterpret_cast<char*>(&gt_dim), 4);
        gt_in.seekg(0, std::ios::end);
        std::ios::pos_type ss = gt_in.tellg();
        size_t fsize = static_cast<size_t>(ss);
        gt_in.close();
        std::cout << "load gt from file: " << gt_file << " points_num: " << gt_pts << " dim: " << gt_dim << std::endl;

        const size_t payload = fsize - sizeof(uint32_t) * 2;
        const size_t entries = static_cast<size_t>(gt_pts) * static_cast<size_t>(gt_dim);
        if (payload == entries * sizeof(uint32_t)) {
            efanna2e::load_gt_data<uint32_t>(gt_file.c_str(), gt_pts, gt_dim, gt_ids);
        } else if (payload == entries * (sizeof(uint32_t) + sizeof(float))) {
            efanna2e::load_gt_data_with_dist<uint32_t, float>(gt_file.c_str(), gt_pts, gt_dim, gt_ids, gt_dists);
        } else {
            std::cerr << "Groundtruth file size mismatch. entries=" << entries
                      << " payload=" << payload << " bytes" << std::endl;
            return -1;
        }
    }

    efanna2e::Metric dist_metric = efanna2e::INNER_PRODUCT;
    if (dist == "l2") {
        dist_metric = efanna2e::L2;
        std::cout << "Using l2 as distance metric" << std::endl;
    } else if (dist == "ip") {
        dist_metric = efanna2e::INNER_PRODUCT;
        std::cout << "Using inner product as distance metric" << std::endl;
    } else if (dist == "cosine") {
        dist_metric = efanna2e::COSINE;
        std::cout << "Using cosine as distance metric" << std::endl;
    } else {
        std::cout << "Unknown distance type: " << dist << std::endl;
        return -1;
    }

    if (!std::filesystem::exists(projection_index_save_file.c_str())) {
        std::cout << "projection index file does not exist." << std::endl;
        return -1;
    }

    efanna2e::IndexBipartite index(q_dim, base_num + sq_num, dist_metric, nullptr);

    index.LoadSearchNeededData(base_data_file.c_str(), sampled_query_data_file.c_str());

    std::cout << "Load graph index: " << projection_index_save_file << std::endl;
    index.LoadProjectionGraph(projection_index_save_file.c_str());

    if (index.need_normalize) {
        std::cout << "Normalizing query data" << std::endl;
        for (uint32_t i = 0; i < q_pts; i++) {
            efanna2e::normalize<float>(aligned_query_data + i * q_dim, q_dim);
        }
    }

    index.InitVisitedListPool(num_threads);

    uint32_t *res = new uint32_t[q_pts * k];
    memset(res, 0, sizeof(uint32_t) * q_pts * k);

    std::vector<std::vector<float>> res_dists(q_pts, std::vector<float>(k, 0.0f));

    uint32_t *projection_cmps_vec = (uint32_t *)aligned_alloc(4, sizeof(uint32_t) * q_pts);
    memset(projection_cmps_vec, 0, sizeof(uint32_t) * q_pts);

    uint32_t *hops_vec = (uint32_t *)aligned_alloc(4, sizeof(uint32_t) * q_pts);
    memset(hops_vec, 0, sizeof(uint32_t) * q_pts);

    uint32_t *jump_mid_vec = (uint32_t *)aligned_alloc(4, sizeof(uint32_t) * q_pts);
    memset(jump_mid_vec, 0, sizeof(uint32_t) * q_pts);

    uint32_t *non_jump_mid_vec = (uint32_t *)aligned_alloc(4, sizeof(uint32_t) * q_pts);
    memset(non_jump_mid_vec, 0, sizeof(uint32_t) * q_pts);

    float *projection_latency_vec = (float *)aligned_alloc(4, sizeof(float) * q_pts);
    memset(projection_latency_vec, 0, sizeof(float) * q_pts);

    std::vector<double> per_query_latency_ms(q_pts, 0.0);

    std::ofstream evaluation_out;
    if (!evaluation_save_path.empty()) {
        evaluation_out.open(evaluation_save_path, std::ios::out);
    }

    std::ofstream per_query_out;
    if (!per_query_csv_path.empty()) {
        per_query_out.open(per_query_csv_path, std::ios::out);
        if (per_query_out.is_open()) {
            per_query_out << "L_pq,query_id,recall@1,recall@10,recall@100,latency_ms,qps" << std::endl;
        }
    }

    std::ofstream per_query_log_out;
    per_query_log_out.open(per_query_log_path, std::ios::out);
    if (per_query_log_out.is_open()) {
        per_query_log_out << "# per-query roargraph search statistics" << std::endl;
        per_query_log_out << "# format: search_mode L_pq query_id distance_cmps hops jump_mid_count non_jump_mid_count latency_ms" << std::endl;
    }

    uint32_t total_distance_computations = 0;

    for (uint32_t L_pq : L_vec) {
        if (k > L_pq) {
            std::cout << "L_pq must greater or equal than k" << std::endl;
            exit(1);
        }

        parameters.Set<uint32_t>("L_pq", L_pq);
        parameters.Set<bool>("enable_jump_search", search_mode == "jump");
        parameters.Set<uint32_t>("max_jump_candidates", max_jump_candidates);

        for (size_t i = 0; i < std::min<size_t>(100, q_pts); ++i) {
            index.SearchRoarGraph(aligned_query_data + i * q_dim, k, i, parameters, res + i * k, res_dists[i]);
        }

        auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1) reduction(+:total_distance_computations)
        for (size_t i = 0; i < q_pts; ++i) {
            auto q_start = std::chrono::high_resolution_clock::now();

            auto ret_val = index.SearchRoarGraph(aligned_query_data + i * q_dim, k, i, parameters,
                                                 res + i * k, res_dists[i]);

            auto q_end = std::chrono::high_resolution_clock::now();

            projection_cmps_vec[i] = ret_val.distance_cmps;
            hops_vec[i] = ret_val.hops;
            jump_mid_vec[i] = ret_val.jump_mid_count;
            non_jump_mid_vec[i] = ret_val.non_jump_mid_count;

            const double latency_ms =
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(q_end - q_start).count();
            per_query_latency_ms[i] = latency_ms;
            projection_latency_vec[i] = static_cast<float>(latency_ms);

            total_distance_computations += ret_val.distance_cmps;
        }

        auto end = std::chrono::high_resolution_clock::now();

        if (per_query_log_out.is_open()) {
            per_query_log_out << "===== search_mode=" << search_mode
                              << " L_pq=" << L_pq
                              << " max_jump_candidates=" << max_jump_candidates
                              << " =====" << std::endl;

            for (size_t i = 0; i < q_pts; ++i) {
                per_query_log_out
                    << search_mode << " "
                    << L_pq << " "
                    << i << " "
                    << projection_cmps_vec[i] << " "
                    << hops_vec[i] << " "
                    << jump_mid_vec[i] << " "
                    << non_jump_mid_vec[i] << " "
                    << projection_latency_vec[i]
                    << std::endl;
            }
            per_query_log_out.flush();
        }

        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float qps = (float)q_pts / ((float)diff / 1000.0f);

        float recall = ComputeRecall(q_pts, k, gt_dim, res, gt_ids);
        std::vector<float> recall_per_q = ComputeRecallPerQuery(q_pts, k, gt_dim, res, gt_ids);
        const float recall_p10 = QuantileNearestRank(recall_per_q, 0.10f);
        const float recall_p30 = QuantileNearestRank(recall_per_q, 0.30f);
        const float recall_p50 = QuantileNearestRank(recall_per_q, 0.50f);
        const float recall_p70 = QuantileNearestRank(recall_per_q, 0.70f);
        const float recall_p90 = QuantileNearestRank(recall_per_q, 0.90f);
        const float recall_p99 = QuantileNearestRank(recall_per_q, 0.99f);

        float avg_projection_cmps = 0.0f;
        for (size_t i = 0; i < q_pts; ++i) {
            avg_projection_cmps += projection_cmps_vec[i];
        }
        avg_projection_cmps /= static_cast<float>(q_pts);

        float avg_hops = 0.0f;
        for (size_t i = 0; i < q_pts; ++i) {
            avg_hops += hops_vec[i];
        }
        avg_hops /= static_cast<float>(q_pts);

        float avg_projection_latency = 0.0f;
        for (size_t i = 0; i < q_pts; ++i) {
            avg_projection_latency += projection_latency_vec[i];
        }
        avg_projection_latency /= static_cast<float>(q_pts);

        float avg_jump_mid = 0.0f;
        float avg_non_jump_mid = 0.0f;
        for (size_t i = 0; i < q_pts; ++i) {
            avg_jump_mid += jump_mid_vec[i];
            avg_non_jump_mid += non_jump_mid_vec[i];
        }
        avg_jump_mid /= static_cast<float>(q_pts);
        avg_non_jump_mid /= static_cast<float>(q_pts);

        std::cout << "search_mode: " << search_mode << std::endl;
        std::cout << "max_jump_candidates: " << max_jump_candidates << std::endl;
        std::cout << "总距离计算次数: " << total_distance_computations << std::endl;
        std::cout << "每个查询的平均距离计算次数: "
                  << static_cast<float>(total_distance_computations) / q_pts << std::endl;
        std::cout << "每个查询平均 jump 点数: " << avg_jump_mid << std::endl;
        std::cout << "每个查询平均 non-jump 点数: " << avg_non_jump_mid << std::endl;

        std::cout << L_pq << "\t\t" << qps << "\t\t" << avg_projection_cmps << "\t\t"
                  << ((float)diff / q_pts) << "\t\t" << recall << "\t\t" << recall_p10 << "\t\t"
                  << recall_p30 << "\t\t" << recall_p50 << "\t\t" << recall_p70 << "\t\t"
                  << recall_p90 << "\t\t" << recall_p99 << "\t\t" << avg_hops
                  << std::endl;

        if (evaluation_out.is_open()) {
            evaluation_out << L_pq << "," << qps << "," << avg_projection_cmps << ","
                           << ((float)diff / q_pts) << "," << recall << "," << recall_p10
                           << "," << recall_p30 << "," << recall_p50 << "," << recall_p70
                           << "," << recall_p90 << "," << recall_p99 << "," << avg_hops
                           << std::endl;
        }

        if (per_query_out.is_open()) {
            uint32_t k1 = std::min<uint32_t>(1, k);
            uint32_t k10 = std::min<uint32_t>(10, k);
            uint32_t k100 = std::min<uint32_t>(100, k);

            for (uint32_t i = 0; i < q_pts; ++i) {
                const float r1 = ComputeRecallAtKPerQuery(res + i * k, k, gt_ids + i * gt_dim, gt_dim, k1);
                const float r10 = ComputeRecallAtKPerQuery(res + i * k, k, gt_ids + i * gt_dim, gt_dim, k10);
                const float r100 = ComputeRecallAtKPerQuery(res + i * k, k, gt_ids + i * gt_dim, gt_dim, k100);
                per_query_out << L_pq << "," << i << "," << r1 << "," << r10 << "," << r100
                              << "," << per_query_latency_ms[i] << "," << qps << std::endl;
            }
        }

        total_distance_computations = 0;
    }

    if (evaluation_out.is_open()) {
        evaluation_out.close();
    }
    if (per_query_out.is_open()) {
        per_query_out.close();
    }
    if (per_query_log_out.is_open()) {
        per_query_log_out.close();
    }

    delete[] res;
    free(projection_cmps_vec);
    free(hops_vec);
    free(jump_mid_vec);
    free(non_jump_mid_vec);
    free(projection_latency_vec);

    delete[] aligned_query_data;
    delete[] gt_ids;
    delete[] gt_dists;

    return 0;
}
