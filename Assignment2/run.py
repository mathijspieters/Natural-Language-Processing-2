import os
# BASE RUN
comment = 'base_run'

# for i in range(3):
#     os.system('python3 train_rnnlm.py --comment {}'.format('{}_{}_{}'.format('rnnlm', comment, i)))
#     os.system('python3 train_vae.py --comment {}'.format('{}_{}_{}'.format('vae', comment, i)))

# VAE EXPERIMENTS
# WORD DROPOUT SVAE
kl_annealing = [4, 10]
free_bit = [2]#[0, 2]
word_dropout = [0, 0.1, 0.3, 0.5]
for kl in kl_annealing:
    for fb in free_bit:
        for wd in word_dropout:
            cmd = 'python3 train_vae.py --comment {} --annealing_end {} --free_bits {} --word_dropout {}'.format('{}_kl_{}_freebit_{}_word_dropout_{}'.format('svae', kl, fb, wd), kl, fb, wd)
            os.system(cmd)

# RNNLM EXPERIMENTS
num_layers = [2, 3]
num_hidden = [100, 200, 300]
for nl in num_layers:
    for nh in num_hidden:
        cmd = 'python3 train_rnnlm.py --comment {} --num_layers {} --num_hidden {}'.format('{}_num_layers_{}_num_hidden_{}'.format('rnnlm', nl, nh), nl, nh)
        os.system(cmd)

