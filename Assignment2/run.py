import os
# BASE RUN
comment = 'base_run'
os.system('python3 train_rnnlm.py --comment {}'.format('{}_{}'.format('rnnlm', comment)))
os.system('python3 train_vae.py --comment {}'.format('{}_{}'.format('vae', comment)))


# VAE EXPERIMENTS
# WORD DROPOUT SVAE
word_dropout = [0, 0.1, 0.3, 0.5]
for wd in word_dropout:
    cmd = 'python3 train_vae.py --comment {} --word_dropout {}'.format('{}_word_dropout_{}'.format('svae', wd), wd)
    os.system(cmd)

# RNNLM EXPERIMENTS
num_layers = [2, 3, 4]
num_hidden = [100, 200, 300]
for nl in num_layers:
    for nh in num_hidden:
        cmd = 'python3 train_rnnlm.py --comment {} --num_layers {} --num_hidden {}'.format('{}_num_layers_{}_num_hidden_{}'.format('rnnlm', nl, nh), nl, nh)
        os.system(cmd)

