from __future__ import absolute_import

from got10k.experiments import *

from siamfc_VGG_cf_add import TrackerSiamFC
import argparse

if __name__ == '__main__':
    # setup tracker
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    # net_path="./pretrained/siamfc_new/model_e1_BEST.pth"
    args = parser.parse_args()
    net_path = args.model
    tracker = TrackerSiamFC(net_path=net_path)
    
    # setup experiments
    experiments = [
        #ExperimentGOT10k('data/GOT-10k', subset='test'),
        #ExperimentOTB('data/OTB', version=2013),
        ExperimentOTB('/home/user/OTB', version=2015),
        #ExperimentVOT('data/vot2018', version=2018),
        #ExperimentDTB70('data/DTB70'),
        #ExperimentTColor128('data/Temple-color-128'),
        #ExperimentUAV123('data/UAV123', version='UAV123'),
        #ExperimentUAV123('data/UAV123', version='UAV20L'),
        #ExperimentNfS('data/nfs', fps=30),
        #ExperimentNfS('data/nfs', fps=240)
    ]

    # run tracking experiments and report performance
    for e in experiments:
        e.run(tracker, visualize=False)
        e.report([tracker.name])
