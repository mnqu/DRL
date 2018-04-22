#include "linelib.h"

sampler::sampler()
{
    n = 0;
    alias = 0;
    prob = 0;
}

sampler::~sampler()
{
    n = 0;
    if (alias != NULL) { free(alias); alias = NULL; }
    if (prob != NULL) { free(prob); prob = NULL; }
}

void sampler::init(long long ndata, double *p)
{
    n = ndata;
    
    alias = (long long *)malloc(n * sizeof(long long));
    prob = (double *)malloc(n * sizeof(double));
    
    long long i, a, g;
    
    // Local workspace:
    double *P;
    long long *S, *L;
    P = (double *)malloc(n * sizeof(double));
    S = (long long *)malloc(n * sizeof(long long));
    L = (long long *)malloc(n * sizeof(long long));
    
    // Normalise given probabilities:
    double sum = 0;
    for (i = 0; i < n; ++i)
    {
        if (p[i] < 0)
        {
            fprintf(stderr, "ransampl: invalid probability p[%d]<0\n", (int)(i));
            exit(1);
        }
        sum += p[i];
    }
    if (!sum)
    {
        fprintf(stderr, "ransampl: no nonzero probability\n");
        exit(1);
    }
    for (i = 0; i < n; ++i) P[i] = p[i] * n / sum;
    
    // Set separate index lists for small and large probabilities:
    long long nS = 0, nL = 0;
    for (i = n - 1; i >= 0; --i)
    {
        // at variance from Schwarz, we revert the index order
        if (P[i] < 1)
            S[nS++] = i;
        else
            L[nL++] = i;
    }
    
    // Work through index lists
    while (nS && nL)
    {
        a = S[--nS]; // Schwarz's l
        g = L[--nL]; // Schwarz's g
        prob[a] = P[a];
        alias[a] = g;
        P[g] = P[g] + P[a] - 1;
        if (P[g] < 1)
            S[nS++] = g;
        else
            L[nL++] = g;
    }
    
    while (nL) prob[L[--nL]] = 1;
    
    while (nS) prob[S[--nS]] = 1;
    
    free(P);
    free(S);
    free(L);
}

long long sampler::draw(double ran1, double ran2)
{
    long long i = n * ran1;
    return ran2 < prob[i] ? i : alias[i];
}

line_node::line_node() : vec(NULL, 0, 0)
{
    node = NULL;
    node_size = 0;
    node_max_size = 1000;
    vector_size = 0;
    node_file[0] = 0;
    node_hash = NULL;
    _vec = NULL;
}

line_node::~line_node()
{
    if (node != NULL) {free(node); node = NULL;}
    node_size = 0;
    node_max_size = 0;
    vector_size = 0;
    node_file[0] = 0;
    if (node_hash != NULL) {free(node_hash); node_hash = NULL;}
    if (_vec != NULL) {free(_vec); _vec = NULL;}
    new (&vec) Eigen::Map<BLPMatrix>(NULL, 0, 0);
}

int line_node::get_hash(char *word)
{
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % hash_table_size;
    return hash;
}

int line_node::search(char *word)
{
    unsigned int hash = get_hash(word);
    while (1) {
        if (node_hash[hash] == -1) return -1;
        if (!strcmp(word, node[node_hash[hash]].word)) return node_hash[hash];
        hash = (hash + 1) % hash_table_size;
    }
    return -1;
}

int line_node::add_node(char *word)
{
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    node[node_size].word = (char *)calloc(length, sizeof(char));
    strcpy(node[node_size].word, word);
    node_size++;
    // Reallocate memory if needed
    if (node_size + 2 >= node_max_size) {
        node_max_size += 1000;
        node = (struct struct_node *)realloc(node, node_max_size * sizeof(struct struct_node));
    }
    hash = get_hash(word);
    while (node_hash[hash] != -1) hash = (hash + 1) % hash_table_size;
    node_hash[hash] = node_size - 1;
    return node_size - 1;
}

void line_node::init(const char *file_name, int vector_dim)
{
    strcpy(node_file, file_name);
    vector_size = vector_dim;
    
    node = (struct struct_node *)calloc(node_max_size, sizeof(struct struct_node));
    node_hash = (int *)calloc(hash_table_size, sizeof(int));
    for (int k = 0; k != hash_table_size; k++) node_hash[k] = -1;
    
    FILE *fi = fopen(node_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: node file not found!\n");
        printf("%s\n", node_file);
        exit(1);
    }
    
    char word[MAX_STRING];
    node_size = 0;
    while (1)
    {
        if (fscanf(fi, "%s", word) != 1) break;
        add_node(word);
    }
    fclose(fi);
    
    long long a, b;
    a = posix_memalign((void **)&_vec, 128, (long long)node_size * vector_size * sizeof(real));
    if (_vec == NULL) { printf("Memory allocation failed\n"); exit(1); }
    for (b = 0; b < vector_size; b++) for (a = 0; a < node_size; a++)
        _vec[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    new (&vec) Eigen::Map<BLPMatrix>(_vec, node_size, vector_size);
    
    printf("Reading nodes from file: %s, DONE!\n", node_file);
    printf("Node size: %d\n", node_size);
    printf("Node dims: %d\n", vector_size);
}

void line_node::output(const char *file_name, int binary)
{
    FILE *fo = fopen(file_name, "wb");
    fprintf(fo, "%d %d\n", node_size, vector_size);
    for (int a = 0; a != node_size; a++)
    {
        fprintf(fo, "%s ", node[a].word);
        if (binary) for (int b = 0; b != vector_size; b++) fwrite(&_vec[a * vector_size + b], sizeof(real), 1, fo);
        else for (int b = 0; b != vector_size; b++) fprintf(fo, "%lf ", _vec[a * vector_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

int line_node::get_node_size()
{
    return node_size;
}

int line_node::get_vector_size()
{
    return vector_size;
}

struct struct_node *line_node::get_node()
{
    return node;
}

real *line_node::get_vec()
{
    return _vec;
}

line_hin::line_hin()
{
    hin_file[0] = 0;
    node_u = NULL;
    node_v = NULL;
    hin = NULL;
    hin_size = 0;
}

line_hin::~line_hin()
{
    hin_file[0] = 0;
    node_u = NULL;
    node_v = NULL;
    if (hin != NULL) {delete [] hin; hin = NULL;}
    hin_size = 0;
}

void line_hin::init(const char *file_name, line_node *p_u, line_node *p_v, bool with_type)
{
    strcpy(hin_file, file_name);
    
    node_u = p_u;
    node_v = p_v;
    
    int node_size = node_u->node_size;
    hin = new std::vector<hin_nb>[node_size];
    
    FILE *fi = fopen(hin_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: hin file not found!\n");
        printf("%s\n", hin_file);
        exit(1);
    }
    char word1[MAX_STRING], word2[MAX_STRING];
    int u, v, tp;
    double w;
    hin_nb curnb;
    if (with_type)
    {
        while (fscanf(fi, "%s %s %lf %d", word1, word2, &w, &tp) == 4)
        {
            if (hin_size % 10000 == 0)
            {
                printf("%lldK%c", hin_size / 1000, 13);
                fflush(stdout);
            }
            
            u = node_u->search(word1);
            v = node_v->search(word2);
            
            if (u != -1 && v != -1)
            {
                curnb.nb_id = v;
                curnb.eg_tp = tp;
                curnb.eg_wei = w;
                hin[u].push_back(curnb);
                hin_size++;
            }
        }
    }
    else
    {
        while (fscanf(fi, "%s %s %lf", word1, word2, &w) == 3)
        {
            if (hin_size % 10000 == 0)
            {
                printf("%lldK%c", hin_size / 1000, 13);
                fflush(stdout);
            }
            
            u = node_u->search(word1);
            v = node_v->search(word2);
            tp = 0;
            
            if (u != -1 && v != -1)
            {
                curnb.nb_id = v;
                curnb.eg_tp = tp;
                curnb.eg_wei = w;
                hin[u].push_back(curnb);
                hin_size++;
            }
        }
    }
    fclose(fi);
    
    printf("Reading edges from file: %s, DONE!\n", hin_file);
    printf("Edge size: %lld\n", hin_size);
}

line_adjacency::line_adjacency()
{
    adjmode = 1;
    edge_tp = 0;
    u_wei = NULL;
    u_nb_cnt = NULL;
    u_nb_id = NULL;
    u_nb_wei = NULL;
    smp_u_nb = NULL;
    v_nb_cnt = NULL;
    v_nb_id = NULL;
    v_nb_wei = NULL;
    smp_v_nb = NULL;
}

line_adjacency::~line_adjacency()
{
    adjmode = 1;
    edge_tp = 0;
    if (u_wei != NULL) {free(u_wei); u_wei = NULL;}
    if (u_nb_cnt != NULL) {free(u_nb_cnt); u_nb_cnt = NULL;}
    if (u_nb_id != NULL) {free(u_nb_id); u_nb_id = NULL;}
    if (u_nb_wei != NULL) {free(u_nb_wei); u_nb_wei = NULL;}
    if (v_nb_cnt != NULL) {free(v_nb_cnt); v_nb_cnt = NULL;}
    if (v_nb_id != NULL) {free(v_nb_id); v_nb_id = NULL;}
    if (v_nb_wei != NULL) {free(v_nb_wei); v_nb_wei = NULL;}
    if (smp_u_nb != NULL)
    {
        delete[] smp_u_nb;
        smp_u_nb = NULL;
    }
    if (smp_v_nb != NULL)
    {
        delete[] smp_v_nb;
        smp_v_nb = NULL;
    }
}

void line_adjacency::init(line_hin *p_hin, int edge_type, int mode)
{
    phin = p_hin;
    adjmode = mode;
    edge_tp = edge_type;
    
    line_node *node_u = phin->node_u, *node_v = phin->node_v;
    
    long long adj_size = 0;
    
    // compute the degree of vertices
    u_wei = (double *)calloc(node_u->node_size, sizeof(double));
    u_nb_cnt = (int *)calloc(node_u->node_size, sizeof(int));
    v_nb_cnt = (int *)calloc(node_v->node_size, sizeof(int));
    double *u_len = (double *)calloc(node_u->node_size, sizeof(double));
    
    for (int u = 0; u != node_u->node_size; u++)
    {
        for (int k = 0; k != (int)(phin->hin[u].size()); k++)
        {
            int v = phin->hin[u][k].nb_id;
            int cur_edge_type = phin->hin[u][k].eg_tp;
            double wei = phin->hin[u][k].eg_wei;
            
            if (cur_edge_type != edge_tp) continue;
            
            u_wei[u] += wei;
            u_nb_cnt[u] += 1;
            v_nb_cnt[v] += 1;
            if (adjmode == 21) u_len[u] += wei;
            if (adjmode == 22) u_len[u] += wei * wei;
            
            adj_size += 1;
        }
    }
    
    if (adjmode != 1) for (int u = 0; u != node_u->node_size; u++)
    {
        if (u_nb_cnt[u] == 0) u_wei[u] = 0;
        else u_wei[u] = 1;
    }
    
    smp_u.init(node_u->node_size, u_wei);
    
    if (adjmode == 22) for (int k = 0; k != node_u->node_size; k++)
    {
        if (u_len[k] != 0) u_len[k] = sqrt(u_len[k]);
        else u_len[k] = 1;
    }
    
    u_nb_id = (int **)malloc(node_u->node_size * sizeof(int *));
    u_nb_wei = (double **)malloc(node_u->node_size * sizeof(double *));
    for (int k = 0; k != node_u->node_size; k++)
    {
        u_nb_id[k] = (int *)malloc(u_nb_cnt[k] * sizeof(int));
        u_nb_wei[k] = (double *)malloc(u_nb_cnt[k] * sizeof(double));
    }
    
    v_nb_id = (int **)malloc(node_v->node_size * sizeof(int *));
    v_nb_wei = (double **)malloc(node_v->node_size * sizeof(double *));
    for (int k = 0; k != node_v->node_size; k++)
    {
        v_nb_id[k] = (int *)malloc(v_nb_cnt[k] * sizeof(int));
        v_nb_wei[k] = (double *)malloc(v_nb_cnt[k] * sizeof(double));
    }
    
    int *pst_u = (int *)calloc(node_u->node_size, sizeof(int));
    int *pst_v = (int *)calloc(node_v->node_size, sizeof(int));
    for (int u = 0; u != node_u->node_size; u++)
    {
        for (int k = 0; k != (int)(phin->hin[u].size()); k++)
        {
            int v = phin->hin[u][k].nb_id;
            int cur_edge_type = phin->hin[u][k].eg_tp;
            double wei = phin->hin[u][k].eg_wei;
            
            if (cur_edge_type != edge_tp) continue;
            
            if (adjmode == 21 || adjmode == 22) wei = wei / u_len[u];
            
            u_nb_id[u][pst_u[u]] = v;
            u_nb_wei[u][pst_u[u]] = wei;
            pst_u[u]++;
            
            v_nb_id[v][pst_v[v]] = u;
            v_nb_wei[v][pst_v[v]] = wei;
            pst_v[v]++;
        }
    }
    free(pst_u);
    free(pst_v);
    free(u_len);
    
    smp_u_nb = new sampler[node_u->node_size];
    for (int k = 0; k != node_u->node_size; k++)
    {
        if (u_nb_cnt[k] == 0) continue;
        smp_u_nb[k].init(u_nb_cnt[k], u_nb_wei[k]);
    }
    
    smp_v_nb = new sampler[node_v->node_size];
    for (int k = 0; k != node_v->node_size; k++)
    {
        if (v_nb_cnt[k] == 0) continue;
        smp_v_nb[k].init(v_nb_cnt[k], v_nb_wei[k]);
    }
    
    printf("Reading adjacency from file: %s, DONE!\n", phin->hin_file);
    printf("Adjacency size: %lld\n", adj_size);
}

int line_adjacency::sample(int u, double (*func_rand_num)())
{
    int index, node, v;
    
    if (u == -1) return -1;
    
    if (adjmode == 1)
    {
        if (u_nb_cnt[u] == 0) return -1;
        index = (int)(smp_u_nb[u].draw(func_rand_num(), func_rand_num()));
        node = u_nb_id[u][index];
        return node;
    }
    else
    {
        if (u_nb_cnt[u] == 0) return -1;
        index = (int)(smp_u_nb[u].draw(func_rand_num(), func_rand_num()));
        v = u_nb_id[u][index];
        
        if (v_nb_cnt[v] == 0) return -1;
        index = (int)(smp_v_nb[v].draw(func_rand_num(), func_rand_num()));
        node = v_nb_id[v][index];
        
        return node;
    }
}

int line_adjacency::sample_head(double (*func_rand_num)())
{
    return (int)(smp_u.draw(func_rand_num(), func_rand_num()));
}

line_trainer_line::line_trainer_line()
{
    edge_tp = 0;
    edge_count_actual = 0;
    phin = NULL;
    expTable = NULL;
    u_nb_cnt = NULL;
    u_nb_id = NULL;
    u_nb_wei = NULL;
    u_wei = NULL;
    v_wei = NULL;
    smp_u_nb = NULL;
    expTable = NULL;
    neg_table = NULL;
}

line_trainer_line::~line_trainer_line()
{
    edge_tp = 0;
    edge_count_actual = 0;
    phin = NULL;
    if (expTable != NULL) {free(expTable); expTable = NULL;}
    if (u_nb_cnt != NULL) {free(u_nb_cnt); u_nb_cnt = NULL;}
    if (u_nb_id != NULL) {free(u_nb_id); u_nb_id = NULL;}
    if (u_nb_wei != NULL) {free(u_nb_wei); u_nb_wei = NULL;}
    if (u_wei != NULL) {free(u_wei); u_wei = NULL;}
    if (v_wei != NULL) {free(v_wei); v_wei = NULL;}
    if (smp_u_nb != NULL)
    {
        delete[] smp_u_nb;
        smp_u_nb = NULL;
    }
    if (neg_table != NULL) {free(neg_table); neg_table = NULL;}
}

void line_trainer_line::init(line_hin *p_hin, int edge_type)
{
    edge_tp = edge_type;
    phin = p_hin;
    line_node *node_u = phin->node_u, *node_v = phin->node_v;
    if (node_u->vector_size != node_v->vector_size)
    {
        printf("ERROR: vector dimsions are not same!\n");
        exit(1);
    }
    
    // compute the degree of vertices
    u_nb_cnt = (int *)calloc(node_u->node_size, sizeof(int));
    u_wei = (double *)calloc(node_u->node_size, sizeof(double));
    v_wei = (double *)calloc(node_v->node_size, sizeof(double));
    for (int u = 0; u != node_u->node_size; u++)
    {
        for (int k = 0; k != (int)(phin->hin[u].size()); k++)
        {
            int v = phin->hin[u][k].nb_id;
            int cur_edge_type = phin->hin[u][k].eg_tp;
            double wei = phin->hin[u][k].eg_wei;
            
            if (cur_edge_type != edge_tp) continue;
            
            u_nb_cnt[u]++;
            u_wei[u] += wei;
            v_wei[v] += wei;
        }
    }
    
    // allocate spaces for edges
    u_nb_id = (int **)malloc(node_u->node_size * sizeof(int *));
    u_nb_wei = (double **)malloc(node_u->node_size * sizeof(double *));
    for (int k = 0; k != node_u->node_size; k++)
    {
        u_nb_id[k] = (int *)malloc(u_nb_cnt[k] * sizeof(int));
        u_nb_wei[k] = (double *)malloc(u_nb_cnt[k] * sizeof(double));
    }
    
    // read neighbors
    int *pst = (int *)calloc(node_u->node_size, sizeof(int));
    for (int u = 0; u != node_u->node_size; u++)
    {
        for (int k = 0; k != (int)(phin->hin[u].size()); k++)
        {
            int v = phin->hin[u][k].nb_id;
            int cur_edge_type = phin->hin[u][k].eg_tp;
            double wei = phin->hin[u][k].eg_wei;
            
            if (cur_edge_type != edge_tp) continue;
            
            u_nb_id[u][pst[u]] = v;
            u_nb_wei[u][pst[u]] = wei;
            pst[u]++;
        }
    }
    free(pst);
    
    // init sampler for edges
    smp_u.init(node_u->node_size, u_wei);
    smp_u_nb = new sampler[node_u->node_size];
    for (int k = 0; k != node_u->node_size; k++)
    {
        if (u_nb_cnt[k] == 0) continue;
        smp_u_nb[k].init(u_nb_cnt[k], u_nb_wei[k]);
    }
    
    // Init negative sampling table
    neg_table = (int *)malloc(neg_table_size * sizeof(int));
    
    int a, i;
    double total_pow = 0, d1;
    double power = 0.75;
    for (a = 0; a < node_v->node_size; a++) total_pow += pow(v_wei[a], power);
    a = 0; i = 0;
    d1 = pow(v_wei[i], power) / (double)total_pow;
    while (a < neg_table_size) {
        if ((a + 1) / (double)neg_table_size > d1) {
            i++;
            if (i >= node_v->node_size) {i = node_v->node_size - 1; d1 = 2;}
            d1 += pow(v_wei[i], power) / (double)total_pow;
        }
        else
            neg_table[a++] = i;
    }
    
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
}

void line_trainer_line::copy_neg_table(line_trainer_line *p_trainer_line)
{
    if (phin->node_v->node_size != p_trainer_line->phin->node_v->node_size)
    {
        printf("ERROR: node sizes are not same!\n");
        exit(1);
    }
    
    int node_size = phin->node_v->node_size;
    
    for (int k = 0; k != node_size; k++) v_wei[k] = p_trainer_line->v_wei[k];
    for (int k = 0; k != neg_table_size; k++) neg_table[k] = p_trainer_line->neg_table[k];
}

void line_trainer_line::train_uv(int u, int v, real lr, int neg_samples, real *_error_vec, unsigned long long &rand_index)
{
    int target, label, vector_size;
    real f, g;
    line_node *node_u = phin->node_u, *node_v = phin->node_v;
    
    vector_size = node_u->vector_size;
    Eigen::Map<BLPVector> error_vec(_error_vec, vector_size);
    error_vec.setZero();
    
    for (int d = 0; d < neg_samples + 1; d++)
    {
        if (d == 0)
        {
            target = v;
            label = 1;
        }
        else
        {
            rand_index = rand_index * (unsigned long long)25214903917 + 11;
            target = neg_table[(rand_index >> 16) % neg_table_size];
            if (target == v) continue;
            label = 0;
        }
        f = node_u->vec.row(u) * node_v->vec.row(target).transpose();
        if (f > MAX_EXP) g = (label - 1) * lr;
        else if (f < -MAX_EXP) g = (label - 0) * lr;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * lr;
        error_vec += g * ((node_v->vec.row(target)));
        node_v->vec.row(target) += g * ((node_u->vec.row(u)));
    }
    node_u->vec.row(u) += error_vec;
    new (&error_vec) Eigen::Map<BLPMatrix>(NULL, 0, 0);
}

void line_trainer_line::train_sample_thread(int tid, long long samples, int negative, real lr, double (*func_rand_num)(), int threads)
{
    int u, v, index;
    unsigned long long rand_index = tid;
    int vector_size = phin->node_u->vector_size;
    real *_error_vec = (real *)malloc(vector_size * sizeof(real));
    long long edge_count = 0, last_edge_count = 0;
    
    while (1)
    {
        //judge for exit
        if (edge_count > samples / threads + 2) break;
        
        if (edge_count - last_edge_count > 1000)
        {
            edge_count_actual += edge_count - last_edge_count;
            last_edge_count = edge_count;
            printf("%cTrainer LINE Progress: %.3lf%%", 13, (real)edge_count_actual / (real)(samples + 1) * 100);
            fflush(stdout);
        }
        
        u = (int)(smp_u.draw(func_rand_num(), func_rand_num()));
        if (u_nb_cnt[u] == 0) return;
        index = (int)(smp_u_nb[u].draw(func_rand_num(), func_rand_num()));
        v = u_nb_id[u][index];
        
        train_uv(u, v, lr, negative, _error_vec, rand_index);
        
        edge_count += 1;
    }
    free(_error_vec);
}

void *line_trainer_line::train_sample_thread_caller(void *arg)
{
    line_trainer_line *pt = (line_trainer_line *)(((arg_struct *)arg)->pt);
    int tid = ((arg_struct *)arg)->tid;
    long long samples = ((arg_struct *)arg)->samples;
    int negative = ((arg_struct *)arg)->negative;
    real lr = ((arg_struct *)arg)->lr;
    double (*func_rand_num)() = ((arg_struct *)arg)->func_rand_num;
    int threads = ((arg_struct *)arg)->threads;
    
    pt->train_sample_thread(tid, samples, negative, lr, func_rand_num, threads);
    pthread_exit(NULL);
}

void line_trainer_line::train_sample(long long samples, int negative, real lr, double (*func_rand_num)(), int threads)
{
    edge_count_actual = 0;
    pthread_t *pt = new pthread_t[threads];
    for (int k = 0; k < threads; k++) pthread_create(&pt[k], NULL, line_trainer_line::train_sample_thread_caller, new arg_struct(this, k, samples, lr, negative, func_rand_num, threads, 0, NULL, 0));
    for (int k = 0; k < threads; k++) pthread_join(pt[k], NULL);
    delete[] pt;
    //printf("\n");
}

void line_trainer_line::train_sample_depth_thread(int tid, long long samples, int negative, real lr, double (*func_rand_num)(), int threads, int depth, line_adjacency *p_adjacency, char pst)
{
    int u, v, index;
    unsigned long long rand_index = tid;
    int vector_size = phin->node_u->vector_size;
    real *_error_vec = (real *)malloc(vector_size * sizeof(real));
    long long edge_count = 0, last_edge_count = 0;
    std::vector<int> node_lst;
    
    while (1)
    {
        //judge for exit
        if (edge_count > samples / threads + 2) break;
        
        if (edge_count - last_edge_count > 1000)
        {
            edge_count_actual += edge_count - last_edge_count;
            last_edge_count = edge_count;
            printf("%cTrainer LINE Progress: %.3lf%%", 13, (real)edge_count_actual / (real)(samples + 1) * 100);
            fflush(stdout);
        }
        
        node_lst.clear();
        
        u = (int)(smp_u.draw(func_rand_num(), func_rand_num()));
        index = (int)(smp_u_nb[u].draw(func_rand_num(), func_rand_num()));
        v = u_nb_id[u][index];
        
        if (pst == 'r')
        {
            node_lst.push_back(v);
            
            for (int k = 1; k != depth; k++)
            {
                v = p_adjacency->sample(v, func_rand_num);
                node_lst.push_back(v);
            }
            
            for (int k = 0; k != depth; k++)
            {
                v = node_lst[k];
                if (v == -1) continue;
                train_uv(u, v, lr, negative, _error_vec, rand_index);
            }
        }
        else if (pst == 'l')
        {
            node_lst.push_back(u);
            
            for (int k = 1; k != depth; k++)
            {
                u = p_adjacency->sample(u, func_rand_num);
                node_lst.push_back(u);
            }
            
            for (int k = 0; k != depth; k++)
            {
                u = node_lst[k];
                if (u == -1) continue;
                train_uv(u, v, lr, negative, _error_vec, rand_index);
            }
        }
        
        edge_count += depth;
    }
}

void *line_trainer_line::train_sample_depth_thread_caller(void *arg)
{
    line_trainer_line *pt = (line_trainer_line *)(((arg_struct *)arg)->pt);
    int tid = ((arg_struct *)arg)->tid;
    long long samples = ((arg_struct *)arg)->samples;
    int negative = ((arg_struct *)arg)->negative;
    real lr = ((arg_struct *)arg)->lr;
    double (*func_rand_num)() = ((arg_struct *)arg)->func_rand_num;
    int threads = ((arg_struct *)arg)->threads;
    int depth = ((arg_struct *)arg)->depth;
    line_adjacency *p_adjacency = ((arg_struct *)arg)->p_adjacency;
    char pst = ((arg_struct *)arg)->pst;
    
    pt->train_sample_depth_thread(tid, samples, negative, lr, func_rand_num, threads, depth, p_adjacency, pst);
    pthread_exit(NULL);
}

void line_trainer_line::train_sample_depth(long long samples, int negative, real lr, double (*func_rand_num)(), int threads, int depth, line_adjacency *p_adjacency, char pst)
{
    edge_count_actual = 0;
    pthread_t *pt = new pthread_t[threads];
    for (int k = 0; k < threads; k++) pthread_create(&pt[k], NULL, line_trainer_line::train_sample_depth_thread_caller, new arg_struct(this, k, samples, lr, negative, func_rand_num, threads, depth, p_adjacency, pst));
    for (int k = 0; k < threads; k++) pthread_join(pt[k], NULL);
    delete[] pt;
    //printf("\n");
}


// **************************************************

line_node_classifier::line_node_classifier()
{
    node = NULL;
    wei = NULL;
    neu0 = NULL;
    neu1 = NULL;
    vector_size = 0;
    label_size = 0;
    train_size = 0;
    test_size = 0;
    train_set.clear();
    test_set.clear();
}

line_node_classifier::~line_node_classifier()
{
    node = NULL;
    if (wei != NULL) {free(wei); wei = NULL;}
    if (neu0 != NULL) {free(neu0); neu0 = NULL;}
    if (neu1 != NULL) {free(neu1); neu1 = NULL;}
    vector_size = 0;
    label_size = 0;
    train_size = 0;
    test_size = 0;
    train_set.clear();
    test_set.clear();
}

void line_node_classifier::init(line_node *pnode, const char *train_file, const char *test_file)
{
    node = pnode;
    vector_size = node->get_vector_size();
    
    FILE *fi;
    char word[MAX_STRING];
    int wid, lid;
    Instance ins;
    
    label_size = 0;
    train_size = 0;
    test_size = 0;
    
    fi = fopen(train_file, "rb");
    while (fscanf(fi, "%s %d", word, &lid) == 2)
    {
        wid = node->search(word);
        if (wid == -1) continue;
        
        if (lid >= label_size) label_size = lid + 1;
        ins.wid = wid;
        ins.lid = lid;
        train_set.push_back(ins);
        train_size++;
    }
    fclose(fi);

    std::random_shuffle(train_set.begin(), train_set.end());
    
    fi = fopen(test_file, "rb");
    while (fscanf(fi, "%s %d", word, &lid) == 2)
    {
        wid = node->search(word);
        if (wid == -1) continue;
        
        if (lid >= label_size) label_size = lid + 1;
        ins.wid = wid;
        ins.lid = lid;
        test_set.push_back(ins);
        test_size++;
    }
    fclose(fi);
    
    printf("Reading classifier data from file: %s %s, DONE!\n", train_file, test_file);
    printf("Label size: %d\n", label_size);
    printf("Train size: %d\n", train_size);
    printf("Test size: %d\n", test_size);
    
    wei = (real *)malloc(vector_size * label_size * sizeof(real));
    for (int a = 0; a != label_size; a++) for (int b = 0; b != vector_size; b++)
        wei[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    neu0 = (real *)malloc(vector_size * sizeof(real));
    neu1 = (real *)malloc(vector_size * sizeof(real));
}

real line_node_classifier::train(int iters, real lr)
{
    real sum, prec = 0, last_prec = 0;
    int wid, lid, nprec, cnt = 0;
    real *vec = node->get_vec();
    
    for (int a = 0; a != label_size; a++) for (int b = 0; b != vector_size; b++)
        wei[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    
    while (lr > 0.0001 && cnt < iters)
    {
        cnt++;
        
        nprec = 0;
        for (int k = 0; k != train_size; k++)
        {
            wid = train_set[k].wid;
            lid = train_set[k].lid;
            sum = 0;
            for (int c = 0; c != vector_size; c++) neu0[c] = vec[wid * vector_size + c];
            
            for (int l = 0; l != label_size; l++)
            {
                neu1[l] = 0;
                for (int c = 0; c != vector_size; c++) neu1[l] += neu0[c] * wei[l * vector_size + c];
                neu1[l] = exp(neu1[l]);
                sum += neu1[l];
            }
            
            int flag = 1;
            for (int l = 0; l != label_size; l++) if (neu1[l] > neu1[lid]) flag = 0;
            if (flag == 1) nprec++;
            
            for (int l = 0; l != label_size; l++) neu1[l] /= sum;
            for (int l = 0; l != label_size; l++) neu1[l] = -neu1[l];
            neu1[lid]++;
            for (int l = 0; l != label_size; l++) for (int c = 0; c != vector_size; c++)
                wei[l * vector_size + c] += lr * neu1[l] * neu0[c];
        }
        
        prec = (real)(nprec) / train_size;
        if (prec < last_prec) lr /= 2;
        last_prec = prec;
    }
    return prec;
}

real line_node_classifier::test()
{
    real prec = 0;
    int wid, lid, nprec;
    real *vec = node->get_vec();
    
    nprec = 0;
    for (int k = 0; k != test_size; k++)
    {
        wid = test_set[k].wid;
        lid = test_set[k].lid;
        for (int c = 0; c != vector_size; c++) neu0[c] = vec[wid * vector_size + c];
        for (int l = 0; l != label_size; l++)
        {
            neu1[l] = 0;
            for (int c = 0; c != vector_size; c++) neu1[l] += neu0[c] * wei[l * vector_size + c];
        }
        int flag = 1;
        for (int l = 0; l != label_size; l++) if (neu1[l] > neu1[lid]) flag = 0;
        if (flag == 1) nprec++;
    }
    
    prec = (real)(nprec) / test_size;
    return prec;
}

line_emb_backup::line_emb_backup()
{
    node = NULL;
    pt = NULL;
    size = 0;
}

line_emb_backup::~line_emb_backup()
{
    node = NULL;
    if (pt != NULL) {free(pt); pt = NULL;}
    size = 0;
}

void line_emb_backup::init(line_node *pnode)
{
    node = pnode;
    long long vector_size = node->get_vector_size();
    long long node_size = node->get_node_size();
    size = vector_size * node_size;
    pt = (real *)malloc(size * sizeof(real));
    //printf("Creating embedding backup %lld, DONE!\n", size);
}

void line_emb_backup::save_in()
{
    real *vec = node->get_vec();
    for (long long k = 0; k != size; k++) pt[k] = vec[k];
}

void line_emb_backup::load_out()
{
    real *vec = node->get_vec();
    for (long long k = 0; k != size; k++) vec[k] = pt[k];
}

