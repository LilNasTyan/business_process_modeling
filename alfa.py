from pm4py.objects.log.importer.xes import importer

from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer

from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator


log = importer.apply("running-example.xes")


# Aлгоритм Alfa
net_alpha, im_alpha, fm_alpha = alpha_miner.apply(log)
visualised_alpha = pn_visualizer.apply(net_alpha, im_alpha, fm_alpha)
pn_visualizer.view(visualised_alpha)


# Метрики качества моделей
fitness_alpha = token_replay.apply(log, net_alpha, im_alpha, fm_alpha)[0]
print(f"Fitness: {fitness_alpha}")

precision_alpha = precision_evaluator.apply(log, net_alpha, im_alpha, fm_alpha)
print(f"Precision: {precision_alpha}")

generalization_alpha = generalization_evaluator.apply(log, net_alpha, im_alpha, fm_alpha)
print(f"Generalization: {generalization_alpha}")