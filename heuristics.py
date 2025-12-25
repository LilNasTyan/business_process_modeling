from pm4py.objects.log.importer.xes import importer

from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.conversion.heuristics_net import converter as hn_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer

from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator


log = importer.apply("running-example.xes")


# Aлгоритм Heuristics
heu_net = heuristics_miner.apply_heu(log)
net_hn, im_hn, fm_hn = hn_converter.apply(heu_net)
visualised_heuristics = pn_visualizer.apply(net_hn, im_hn, fm_hn)
pn_visualizer.view(visualised_heuristics)


# Метрики качества моделей
fitness_heuristics = token_replay.apply(log, net_hn, im_hn, fm_hn)[0]
print(f"Fitness: {fitness_heuristics}")

precision_heuristics = precision_evaluator.apply(log, net_hn, im_hn, fm_hn)
print(f"Precision: {precision_heuristics}")

generalization_heuristics = generalization_evaluator.apply(log, net_hn, im_hn, fm_hn)
print(f"Generalization: {generalization_heuristics}")
