#include <Python.h>
#include "linelib.h"
#include <gsl/gsl_rng.h>
#include <vector>

std::vector<line_node *> vec_pnode;
std::vector<line_hin *> vec_phin;
std::vector<line_adjacency *> vec_padjacency;
std::vector<line_trainer_line *> vec_ptrainer_line;
std::vector<line_node_classifier *> vec_pclassifier;
std::vector<line_emb_backup *> vec_pbackup;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

double func_rand_num()
{
    return gsl_rng_uniform(gsl_r);
}

static PyObject *add_node(PyObject *self, PyObject *args)
{
    char *file_name;
    int vector_size;
    
    if (!PyArg_ParseTuple(args, "s|i", &file_name, &vector_size))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    line_node *pnode = new line_node;
    pnode->init(file_name, vector_size);
    long id = (long)(vec_pnode.size());
    vec_pnode.push_back(pnode);
    return PyInt_FromLong(id);
}

static PyObject *get_node_id(PyObject *self, PyObject *args)
{
    int nid;
    char *node;
    
    if (!PyArg_ParseTuple(args, "i|s", &nid, &node))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    int id = vec_pnode[nid]->search(node);
    return PyInt_FromLong(id);
}

static PyObject *get_node_name(PyObject *self, PyObject *args)
{
    int nid, id;
    
    if (!PyArg_ParseTuple(args, "i|i", &nid, &id))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    return PyString_FromString(vec_pnode[nid]->get_node()[id].word);
}

static PyObject *get_node_size(PyObject *self, PyObject *args)
{
    int nid;
    
    if (!PyArg_ParseTuple(args, "i", &nid))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    long size = vec_pnode[nid]->get_node_size();
    return PyInt_FromLong(size);
}

static PyObject *get_node_dims(PyObject *self, PyObject *args)
{
    int nid;
    
    if (!PyArg_ParseTuple(args, "i", &nid))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    long size = vec_pnode[nid]->get_vector_size();
    return PyInt_FromLong(size);
}

static PyObject *get_node_vecs(PyObject *self, PyObject *args)
{
    int nid;
    
    if (!PyArg_ParseTuple(args, "i", &nid))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    int node_size = vec_pnode[nid]->get_node_size();
    int vector_size = vec_pnode[nid]->get_vector_size();
    struct struct_node *node = vec_pnode[nid]->get_node();
    real *vec = vec_pnode[nid]->get_vec();
    
    PyObject* result = PyList_New(0);
    for (int k = 0; k != node_size; k++)
    {
        PyObject* curvec = PyList_New(0);
        for (int c = 0; c != vector_size; c++)
        {
            double f = vec[k * vector_size + c];
            PyList_Append(curvec, PyFloat_FromDouble(f));
        }
        
        PyObject* name_curvec = PyList_New(0);
        PyList_Append(name_curvec, PyString_FromString(node[k].word));
        PyList_Append(name_curvec, curvec);
        
        PyList_Append(result, name_curvec);
    }
    
    return result;
}

static PyObject *write_node_vecs(PyObject *self, PyObject *args)
{
    int nid, binary;
    char *file_name;
    
    if (!PyArg_ParseTuple(args, "i|s|i", &nid, &file_name, &binary))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    vec_pnode[nid]->output(file_name, binary);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *add_hin(PyObject *self, PyObject *args)
{
    char *file_name;
    int uid, vid, with_type;
    
    if (!PyArg_ParseTuple(args, "s|i|i|i", &file_name, &uid, &vid, &with_type))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    line_hin *phin = new line_hin;
    phin->init(file_name, vec_pnode[uid], vec_pnode[vid], with_type);
    long id = (long)(vec_phin.size());
    vec_phin.push_back(phin);
    return PyInt_FromLong(id);
}

static PyObject *add_adjacency(PyObject *self, PyObject *args)
{
    int nid, edge_type, mode;
    
    if (!PyArg_ParseTuple(args, "i|i|i", &nid, &edge_type, &mode))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    line_adjacency *padj = new line_adjacency;
    padj->init(vec_phin[nid], edge_type, mode);
    long id = (long)(vec_padjacency.size());
    vec_padjacency.push_back(padj);
    return PyInt_FromLong(id);
}

static PyObject *add_trainer_line(PyObject *self, PyObject *args)
{
    int hinid, edge_type;
    
    if (!PyArg_ParseTuple(args, "i|i", &hinid, &edge_type))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    line_trainer_line *ptrainer_line = new line_trainer_line;
    ptrainer_line->init(vec_phin[hinid], edge_type);
    long id = (long)(vec_ptrainer_line.size());
    vec_ptrainer_line.push_back(ptrainer_line);
    return PyInt_FromLong(id);
}

static PyObject *copy_neg_table(PyObject *self, PyObject *args)
{
    int uid, vid;
    
    if (!PyArg_ParseTuple(args, "i|i", &uid, &vid))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    vec_ptrainer_line[uid]->copy_neg_table(vec_ptrainer_line[vid]);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *run_trainer_line(PyObject *self, PyObject *args)
{
    int trainer_id, negative, threads;
    long long samples;
    real lr;
    
    if (!PyArg_ParseTuple(args, "i|L|i|f|i", &trainer_id, &samples, &negative, &lr, &threads))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    vec_ptrainer_line[trainer_id]->train_sample(samples, negative, lr, func_rand_num, threads);
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *run_trainer_line_depth(PyObject *self, PyObject *args)
{
    int trainer_id, negative, threads, adj_id, depth;
    long long samples;
    char pst;
    real lr;
    
    if (!PyArg_ParseTuple(args, "i|L|i|f|i|i|i|c", &trainer_id, &samples, &negative, &lr, &threads, &depth, &adj_id, &pst))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    vec_ptrainer_line[trainer_id]->train_sample_depth(samples, negative, lr, func_rand_num, threads, depth, vec_padjacency[adj_id], pst);
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *add_node_classifier(PyObject *self, PyObject *args)
{
    int nodeid;
    char *train_file, *test_file;
    
    if (!PyArg_ParseTuple(args, "i|s|s", &nodeid, &train_file, &test_file))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    line_node_classifier *pclassifier = new line_node_classifier;
    pclassifier->init(vec_pnode[nodeid], train_file, test_file);
    long id = (long)(vec_pclassifier.size());
    vec_pclassifier.push_back(pclassifier);
    return PyInt_FromLong(id);
}

static PyObject *run_classifier_train(PyObject *self, PyObject *args)
{
    int classid, iters;
    real lr;
    
    if (!PyArg_ParseTuple(args, "i|i|f", &classid, &iters, &lr))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    double prec = vec_pclassifier[classid]->train(iters, lr);
    return PyFloat_FromDouble(prec);
}

static PyObject *run_classifier_test(PyObject *self, PyObject *args)
{
    int classid;
    
    if (!PyArg_ParseTuple(args, "i", &classid))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    double prec = vec_pclassifier[classid]->test();
    return PyFloat_FromDouble(prec);
}

static PyObject *add_emb_backup(PyObject *self, PyObject *args)
{
    int nodeid;
    
    if (!PyArg_ParseTuple(args, "i", &nodeid))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    line_emb_backup *pbackup = new line_emb_backup;
    pbackup->init(vec_pnode[nodeid]);
    long id = (long)(vec_pbackup.size());
    vec_pbackup.push_back(pbackup);
    return PyInt_FromLong(id);
}

static PyObject *save_to(PyObject *self, PyObject *args)
{
    int embid;
    
    if (!PyArg_ParseTuple(args, "i", &embid))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    vec_pbackup[embid]->save_in();
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *load_from(PyObject *self, PyObject *args)
{
    int embid;
    
    if (!PyArg_ParseTuple(args, "i", &embid))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    vec_pbackup[embid]->load_out();
    
    Py_INCREF(Py_None);
    return Py_None;
}

/*  define functions in module */
static PyMethodDef PyExtMethods[] =
{
    //{ "init", init, METH_VARARGS, "init" },
    { "add_node", add_node, METH_VARARGS, "add_node" },
    { "get_node_id", get_node_id, METH_VARARGS, "get_node_id" },
    { "get_node_name", get_node_name, METH_VARARGS, "get_node_name" },
    { "get_node_size", get_node_size, METH_VARARGS, "get_node_size" },
    { "get_node_dims", get_node_dims, METH_VARARGS, "get_node_dims" },
    { "get_node_vecs", get_node_vecs, METH_VARARGS, "get_node_vecs" },
    { "write_node_vecs", write_node_vecs, METH_VARARGS, "write_node_vecs" },
    { "add_hin", add_hin, METH_VARARGS, "add_hin" },
    { "add_adjacency", add_adjacency, METH_VARARGS, "add_adjacency" },
    { "add_trainer_line", add_trainer_line, METH_VARARGS, "add_trainer_line" },
    { "copy_neg_table", copy_neg_table, METH_VARARGS, "copy_neg_table" },
    { "run_trainer_line", run_trainer_line, METH_VARARGS, "run_trainer_line" },
    { "run_trainer_line_depth", run_trainer_line_depth, METH_VARARGS, "run_trainer_line_depth" },
    { "add_node_classifier", add_node_classifier, METH_VARARGS, "add_node_classifier" },
    { "run_classifier_train", run_classifier_train, METH_VARARGS, "run_classifier_train" },
    { "run_classifier_test", run_classifier_test, METH_VARARGS, "run_classifier_test" },
    { "add_emb_backup", add_emb_backup, METH_VARARGS, "add_emb_backup" },
    { "save_to", save_to, METH_VARARGS, "save_to" },
    { "load_from", load_from, METH_VARARGS, "load_from" },
    { NULL, NULL, 0, NULL }
};

/* module initialization */
PyMODINIT_FUNC initpylinelib(void)
{
    (void)Py_InitModule("pylinelib", PyExtMethods);
    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);
}

