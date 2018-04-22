#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <Eigen/Dense>
#include <iostream>

#define MAX_STRING 500
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
const int neg_table_size = 1e8;
const int hash_table_size = 30000000;

typedef float real;

typedef Eigen::Matrix< real, Eigen::Dynamic,
Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign >
BLPMatrix;

typedef Eigen::Matrix< real, 1, Eigen::Dynamic,
Eigen::RowMajor | Eigen::AutoAlign >
BLPVector;

class line_node;
class line_hin;
class line_adjacency;
class line_trainer_line;
class line_trainer_norm;
class line_trainer_reg;
class line_triple;
class line_regularizer_norm;
class line_regularizer_line;

struct struct_node {
    char *word;
};

struct hin_nb {
    int nb_id;
    double eg_wei;
    int eg_tp;
};

struct triple
{
    int h, r, t;
    friend bool operator < (triple t1, triple t2)
    {
        if (t1.h == t2.h)
        {
            if (t1.r == t2.r) return t1.t < t2.t;
            return t1.r < t2.r;
        }
        return t1.h < t2.h;
    }
};

struct arg_struct
{
    void *pt;
    int tid;
    long long samples;
    real lr;
    int negative;
    double (*func_rand_num)();
    int threads;
    int depth;
    line_adjacency *p_adjacency;
    char pst;
    
    arg_struct(void *pt, int tid, long long samples, real lr, int negative, double (*func_rand_num)(), int threads, int depth, line_adjacency *p_adjacency, char pst) : pt(pt), tid(tid), samples(samples), lr(lr), negative(negative), func_rand_num(func_rand_num), threads(threads), depth(depth), p_adjacency(p_adjacency), pst(pst){}
};

class sampler
{
    long long n;
    long long *alias;
    double *prob;
    
public:
    sampler();
    ~sampler();
    
    void init(long long ndata, double *p);
    long long draw(double ran1, double ran2);
};

class line_node
{
protected:
    struct struct_node *node;
    int node_size, node_max_size, vector_size;
    char node_file[MAX_STRING];
    int *node_hash;
    real *_vec;
    Eigen::Map<BLPMatrix> vec;
    
    int get_hash(char *word);
    int add_node(char *word);
public:
    
    line_node();
    ~line_node();
    
    friend class line_hin;
    friend class line_adjacency;
    friend class line_trainer_line;
    friend class line_trainer_norm;
    friend class line_trainer_reg;
    friend class line_triple;
    friend class line_regularizer_norm;
    friend class line_regularizer_line;
    
    void init(const char *file_name, int vector_dim);
    int search(char *word);
    void output(const char *file_name, int binary);
    
    struct struct_node *get_node();
    int get_node_size();
    int get_vector_size();
    real *get_vec();
};

class line_hin
{
protected:
    char hin_file[MAX_STRING];
    
    line_node *node_u, *node_v;
    std::vector<hin_nb> *hin;
    long long hin_size;
    
public:
    line_hin();
    ~line_hin();
    
    friend class line_adjacency;
    friend class line_trainer_line;
    friend class line_trainer_norm;
    friend class line_trainer_reg;
    
    void init(const char *file_name, line_node *p_u, line_node *p_v, bool with_type = 1);
};

class line_adjacency
{
protected:
    line_hin *phin;
    
    int adjmode;
    char edge_tp;
    
    double *u_wei;
    sampler smp_u;
    
    int *u_nb_cnt; int **u_nb_id; double **u_nb_wei;
    sampler *smp_u_nb;
    
    int *v_nb_cnt; int **v_nb_id; double **v_nb_wei;
    sampler *smp_v_nb;
    
public:
    line_adjacency();
    ~line_adjacency();
    
    friend class line_trainer_line;
    friend class line_trainer_norm;
    friend class line_trainer_reg;
    friend class line_regularizer_norm;
    friend class line_regularizer_line;
    
    void init(line_hin *p_hin, int edge_type, int mode);
    int sample(int u, double (*func_rand_num)());
    int sample_head(double (*func_rand_num)());
};

class line_trainer_line
{
protected:
    line_hin *phin;
    
    int *u_nb_cnt; int **u_nb_id; double **u_nb_wei;
    double *u_wei, *v_wei;
    sampler smp_u, *smp_u_nb;
    real *expTable;
    int *neg_table;
    
    int edge_tp;
    long long edge_count_actual;
    
    void train_uv(int u, int v, real lr, int neg_samples, real *_error_vec, unsigned long long &rand_index);
    
    void train_sample_thread(int tid, long long samples, int negative, real lr, double (*func_rand_num)(), int threads);
    static void *train_sample_thread_caller(void *arg);
    
    void train_sample_depth_thread(int tid, long long samples, int negative, real lr, double (*func_rand_num)(), int threads, int depth, line_adjacency *p_adjacency, char pst);
    static void *train_sample_depth_thread_caller(void *arg);
    
public:
    line_trainer_line();
    ~line_trainer_line();
    
    void init(line_hin *p_hin, int edge_type);
    void copy_neg_table(line_trainer_line *p_trainer_line);
    
    void train_sample(long long samples, int negative, real lr, double (*func_rand_num)(), int threads);
    void train_sample_depth(long long samples, int negative, real lr, double (*func_rand_num)(), int threads, int depth, line_adjacency *p_adjacency, char pst);
};

// **************************************************

struct Instance
{
    int wid, lid;
};

class line_node_classifier
{
protected:
    line_node *node;
    real *wei, *neu0, *neu1;
    int vector_size, label_size, train_size, test_size;
    std::vector<Instance> train_set, test_set;
public:
    line_node_classifier();
    ~line_node_classifier();
    
    void init(line_node *pnode, const char *train_file, const char *test_file);
    real train(int iters, real lr);
    real test();
};

class line_emb_backup
{
protected:
    line_node *node;
    long long size;
    real *pt;
public:
    line_emb_backup();
    ~line_emb_backup();
    
    void init(line_node *pnode);
    void save_in();
    void load_out();
};

