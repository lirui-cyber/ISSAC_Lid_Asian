from model.model import *
from model.data_load import *
from score import scoring as scoring
import subprocess 
import json
# data/lre17_eval_3s_10_snrs data/lre17_eval_3s_15_snrs data/lre17_eval_3s_20_snrs data/lre17_eval_10s_5_snrs data/lre17_eval_10s_10_snrs data/lre17_eval_10s_15_snrs data/lre17_eval_10s_20_snrs data/lre17_eval_30s_5_snrs
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def validation(valid_txt, valid_feats, model, model_name, device, kaldi, log_dir, num_lang):
    t_list = [0,0,0,0,0]
    valid_set = RawFeatures2(valid_txt, valid_feats)
    valid_data = DataLoader(dataset=valid_set,
                            batch_size=1,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=collate_fn_atten)
    model.eval()
    correct = 0
    total = 0
    scores = 0
    with torch.no_grad():
        for step, (utt, labels, seq_len) in enumerate(valid_data):
            utt = utt.to(device=device, dtype=torch.float)
            labels = labels.to(device)
            # Forward pas\
            outputs = model(utt, seq_len)
            #outputs_prob = torch.softmax(outputs,dim=-1)
            #prob, _ = torch.max(outputs_prob,-1)
            predicted = torch.argmax(outputs, -1)
            #if prob < 0.5:
            #    predicted = torch.tensor(4) 
            #t_list[int(predicted.item())] += 1
            total += labels.size(-1)
            correct += (predicted == labels).sum().item()
            if step == 0:
                scores = outputs
            else:
                scores = torch.cat((scores, outputs), dim=0)
    acc = correct / total
    #for i in range(len(t_list)):
    #    print("{} {}\n".format(i,t_list[i]))
    print('Current Acc.: {:.2f} %'.format(100 * acc))
    scores = scores.squeeze().cpu().numpy()
    print(scores.shape)
    trial_txt = log_dir + '/trial_{}.txt'.format(model_name)
    score_txt = log_dir + '/score_{}.txt'.format(model_name)
    output_txt = log_dir + '/output_{}.txt'.format(model_name)
    scoring.get_trials(valid_txt, num_lang, trial_txt)
    scoring.get_score(valid_txt, scores, num_lang, score_txt)
    eer_txt = trial_txt.replace('trial', 'eer')
    subprocess.call(f"bash ./score/computeEER.sh "
                    f"--write-file {eer_txt} {trial_txt} {score_txt}", shell=True)
    cavg = scoring.compute_cavg(trial_txt, score_txt)
    print("Cavg: {} ".format(cavg))
    with open(output_txt, 'w') as f:
        f.write("ACC:{} Cavg:{}".format(acc, cavg))
    return cavg


def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--json', type=str, default='xsa_noise_config.json')
    args = parser.parse_args()
    with open(args.json, 'r') as json_obj:
        config_proj= json.load(json_obj)
    seed = config_proj["optim_config"]["seed"]
    if seed == -1:
        pass
    else:
        setup_seed(seed)
    device = torch.device('cuda:{}'.format(config_proj["optim_config"]["device"])
                          if torch.cuda.is_available() else 'cpu')
    feat_dim = config_proj["model_config"]["d_k"]
    n_heads = config_proj["model_config"]["n_heads"]

    if config_proj["check_point"] != None:
        model = X_Transformer_E2E_LID(input_dim=config_proj["model_config"]["feat_dim"],
                                    feat_dim=config_proj["model_config"]["d_k"],
                                    d_k=config_proj["model_config"]["d_k"],
                                    d_v=config_proj["model_config"]["d_k"],
                                    d_ff=config_proj["model_config"]["d_ff"],
                                    n_heads=config_proj["model_config"]["n_heads"],
                                    dropout=0.1,
                                    n_lang=config_proj["model_config"]["n_language"],
                                    max_seq_len=10000)
        model.load_state_dict(torch.load(config_proj["check_point"]))

        model.to(device)

        model_name = config_proj["model_name"]
        log_dir = config_proj["Input"]["userroot"] + config_proj["Input"]["log"]
        kaldi_root = config_proj["kaldi"]
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        feat_layer = config_proj["wav2vec_info"]["layer"]

        # need to change
        test_sets = config_proj["Input"]["test_noise_sets"].split()
        print("## Test set: ", test_sets)
        for test in test_sets:
            # test = "data/lre17_eval_3s_5_snrs"
            print("## Test set: ", test)
            test_txt = config_proj["Input"]["userroot"] + config_proj["Input"]["data"]  + test + "/text"
            test_feats = config_proj["Input"]["userroot"] + config_proj["Input"]["data"] + test +  "/feats.scp"
            logg_dir = log_dir + "/" + test
            if not os.path.exists(logg_dir):
                os.mkdir(logg_dir)
            if test_txt is not None:
                validation(test_txt, test_feats, model, model_name, device, kaldi=kaldi_root, log_dir=logg_dir,
                    num_lang=config_proj["model_config"]["n_language"])
    else:
        print('Model check point for testing is not provided!!!')

if __name__ == "__main__":
    main()
