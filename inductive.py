from pm4py.objects.log.importer.xes import importer

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer

from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator


log = importer.apply("running-example.xes")


# Aлгоритм Inductive
process_tree = inductive_miner.apply(log)
net_ind, im_ind, fm_ind = pt_converter.apply(process_tree, variant=pt_converter.Variants.TO_PETRI_NET)
visualised_inductive = pn_visualizer.apply(net_ind, im_ind, fm_ind)
pn_visualizer.view(visualised_inductive)


# Метрики качества моделей
fitness_inductive = token_replay.apply(log, net_ind, im_ind, fm_ind)[0]
print(f"Fitness: {fitness_inductive}")

precision_inductive = precision_evaluator.apply(log, net_ind, im_ind, fm_ind)
print(f"Precision: {precision_inductive}")

generalization_inductive = generalization_evaluator.apply(log, net_ind, im_ind, fm_ind)
print(f"Generalization: {generalization_inductive}")