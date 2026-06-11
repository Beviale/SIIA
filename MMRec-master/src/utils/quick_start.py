# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os
from common.trainer import MKGATTrainer
import wandb


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)
    # Force wandb online logging regardless of any WANDB_MODE left in the shell
    # (a stale WANDB_MODE=offline/disabled silently stops web logging).
    os.environ['WANDB_MODE'] = 'online'
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])

    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        # build the wandb run name only — the run itself is created right before
        # training (see below), so an aborted/crashed setup doesn't leave an empty
        # "ghost" run online with no data.
        _contrastive = config['use_contrastive'] if config['use_contrastive'] is not None else False
        run_name = f"contrastive{_contrastive}_" + config['dataset'] + "_"
        run_name = run_name + "_".join([f"{j}{k}" for j, k in zip(config['hyper_parameters'], hyper_tuple)])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()

        if config['model_enriched_triples_format'] and config['dataset_support_triplets']:
            train_triplets = train_data.kg_triplets()
            model = get_model(config['model'])(config, train_data, train_triplets).to(config['device'])
        else:
            model = get_model(config['model'])(config, train_data).to(config['device'])

        logger.info(model)

        # trainer loading and initialization
        if config['model'] == "KGAT":
            trainer = MKGATTrainer(config, model, run_name, mg, train_triplets)
        elif config["model"] == "MKGAT":
            trainer = MKGATTrainer(config, model, run_name, mg)
        else:
            trainer = get_trainer()(config, model, mg)
        # model training — create the wandb run only now, when we're about to log.
        # try/finally guarantees the run is closed (and synced) even if fit() raises.
        run = wandb.init(
            project=config['model'],
            name=run_name,
            config={k: config[k] for k in config['hyper_parameters']},
            mode='online',
            reinit=True,
        )
        try:
            best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
                train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        finally:
            run.finish()   # close the run so it syncs and the next init starts clean
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))

