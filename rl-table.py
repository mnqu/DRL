import sys
import os
import random
import struct
import pylinelib as linelib
import mctslib

def search_from_pool(mcnode):
	pointer = mcnode._id
	pst = -1
	for k in range(pool_size):
		if pointer == pointers[k]:
			pst = k
			break
	return pst

def add_to_pool(mcnode):
	pointer = mcnode._id
	pst = search_from_pool(mcnode)
	if pst != -1:
		return pst
	ok = -1
	for pst in range(pool_size):
		if pointers[pst] == -1:
			pointers[pst] = pointer
			linelib.save_to(node_pools[pst])
			linelib.save_to(cont_pools[pst])
			ok = pst
			break
	return ok

def load_from_pool(mcnode):
	pointer = mcnode._id
	pst = search_from_pool(mcnode)
	if pst != -1:
		linelib.load_from(node_pools[pst])
		linelib.load_from(cont_pools[pst])
	return pst

def delete_from_pool(pst):
	if pst >= 0 and pst < pool_size:
		pointers[pst] = -1

def run(curiter, act):
	linelib.run_trainer_line(trainers[act], samples, negative, alpha, threads)
	print '\rIter:', curiter, 'Type:', act, 'Training DONE!'

def calculate_priors():
	return prior_table

def calculate_best():
	bestid, bestvl = -1, -100000000.0
	for act in range(type_size):
		score = prior_table[act]
		if score > bestvl:
			bestid = act
			bestvl = score
	return bestid

cont_file = '../data_dblp/node0.txt'
node_file = '../data_dblp/node1.txt'
net_file = '../data_dblp/hinet.txt'
train_file = '../data_dblp/train.lb'
test_file = '../data_dblp/test.lb'
output_file = 'vec.emb'

vector_size = 100
negative = 5
samples = 1000000
threads = 20
alpha = 0.015
type_size = 3
depth = 5
tree_size = 7
pool_size = 12
lamb = 0.5
binary = 1
penalty = 0.0005
gamma = 0.5

node = linelib.add_node(node_file, vector_size)
cont = linelib.add_node(cont_file, vector_size)
hin = linelib.add_hin(net_file, cont, node, 1)
trainers = [linelib.add_trainer_line(hin, k) for k in range(type_size)]
classifier = linelib.add_node_classifier(node, train_file, test_file)

node_pools = [linelib.add_emb_backup(node) for k in range(pool_size)]
cont_pools = [linelib.add_emb_backup(cont) for k in range(pool_size)]
pointers = [-1 for k in range(pool_size)]

mctree = mctslib.Tree()
mctree.set_lambda(lamb)
mctree.set_action_size(type_size)

prior_table = [random.random() / 10000 for k in range(type_size)]

print 'Training process:'
# set priors for the root node and add some random variance
mctree.get_root().add_prior([random.random() / 100000 for k in range(type_size)])
add_to_pool(mctree.get_root())
stage = 0
while True:
	# add nodes
	current_sample_pool = []
	for T in range(tree_size):
		if mctree.get_size() >= tree_size:
			break
		print '-- Stage:', stage, 'Simu:', T

		# selection
		mcnode = mctree.select()

		# load embeddings
		prev_mcnode = mcnode
		hist_act = []
		# load embeddings from the latest node in the pool
		while prev_mcnode != None:
			if load_from_pool(prev_mcnode) != -1:
				break
			hist_act.insert(0, prev_mcnode.get_last_action())
			prev_mcnode = prev_mcnode.get_parent()
		for k in range(len(prev_mcnode._act)):
			print 'Iter:', stage+k, 'Type:', prev_mcnode._act[k], 'Load from pool!'
		# execute the following actions
		for k in range(len(hist_act)):
			act = hist_act[k]
			run(i + len(prev_mcnode._act) + k, act)

		# find the best action and execute
		action = mcnode.get_max()
		run(stage + len(mcnode._act), action)

		# expand
		next_mcnode = mctree.expand(mcnode, action)
		curvl = linelib.run_classifier_train(classifier, 1, 0.1)
		next_mcnode.set_base(curvl)

		# calculate priors for the new node
		next_mcnode.set_prior(prior_table)

		# save embeddings
		add_to_pool(next_mcnode)

		# simulation
		print '----------'
		if depth != 0:
			lastvl = curvl
			curdepth = 0
			while True:
				simu_action = calculate_best()
				run(stage + len(mcnode._act) + curdepth + 1, simu_action)
				curvl = linelib.run_classifier_train(classifier, 1, 0.1)
				if curvl - lastvl < penalty:
					break
				lastvl = curvl
				curdepth += 1
				if curdepth == depth:
					break
			curvl = lastvl - penalty * curdepth

		# backup
		current_samples = mctree.update_with_penalty(mcnode, action, curvl, penalty)

		# collect data and train RNN
		current_sample_pool += current_samples

	# execute the best action
	print '-- Stage:', stage, 'Final'
	action = mctree.get_root().get_best()
	if mctree.get_root().get_action_score(action) < 0:
		break
	if search_from_pool(mctree.get_root().get_child(action)) == -1:
		load_from_pool(mctree.get_root())
		run(i, action)
	else:
		print 'Iter:', stage, 'Type:', action, 'Load from pool!'
		load_from_pool(mctree.get_root().get_child(action))

	curvl = linelib.run_classifier_train(classifier, 1, 0.1)
	print '!!!!!!!!!!!!!!!!!!!!'
	print 'Iter:', stage, 'Final Decision:', action, curvl
	print '!!!!!!!!!!!!!!!!!!!!'

	# delete other branches
	mctree.derive(action)
	for pst in range(pool_size):
		if mctree.search_id(pointers[pst]) == 0:
			delete_from_pool(pst)
	add_to_pool(mctree.get_root())

	# update prior table
	#print current_sample_pool
	for current_sample in current_sample_pool:
		su, au = current_sample[0][0], current_sample[0][1]
		sv, av = current_sample[1][0], current_sample[1][1]
		reward = current_sample[2]

		if sv == None and av == None:
			prior_table[au] = prior_table[au] + gamma * (reward - prior_table[au])
		else:
			prior_table[au] = prior_table[au] + gamma * (reward + prior_table[av] - prior_table[au])

	print prior_table

	# update stage
	stage += 1

linelib.run_classifier_train(classifier, 100, 0.01)
print 'Test Accuracy:', linelib.run_classifier_test(classifier)
linelib.write_node_vecs(node, output_file, binary)
