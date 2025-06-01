import torch
from torchfunction.device import AutoDevice as device

BATCH_SIZE = 256

def get_cdm_input(bs=BATCH_SIZE, noise_inp=0.1):
    input_tensor = torch.zeros(bs, 61, 4).to(device)
    ctx = torch.randint(2, (bs, )).to(device)
    cnd1 = torch.randint(6, (bs, )).to(device)
    cnd2 = torch.randint(6, (bs, )).to(device)
    cnds = torch.stack([cnd1, cnd2], 1) # [bs, 2]
    cnd_ctx = torch.gather(cnds, 1, ctx.view(-1, 1)).squeeze() # bs
    mask = torch.zeros(bs, 61, 1).to(device)
    mask[:, 60, 0] = 1
    for i in range(bs):
        input_tensor[i, :, 2 + ctx[i]] = 1 # LOC ctx 1 0 FREQ ctx 0 1
        input_tensor[i, 20:60, 0] = [-0.4,-0.2,-0.1,0.1,0.2,0.4][cnd1[i]] + torch.randn(40) * noise_inp
        input_tensor[i, 20:60, 1] = [-0.4,-0.2,-0.1,0.1,0.2,0.4][cnd2[i]] + torch.randn(40) * noise_inp
    y = (2 * (cnd_ctx < 3).float() - 1).view(bs, 1, 1).repeat(1, 61, 1)
    return input_tensor, y, mask, cnds, ctx, cnd_ctx

def get_standard_cdm_input(noise_inp=0.1):
    input_tensor = torch.zeros(72, 61, 4).to(device)
    ctx = list()
    cnd1 = list()
    cnd2 = list()
    for c in range(2):
        for i in range(6):
            for j in range(6):
                ctx.append(c)
                cnd1.append(i)
                cnd2.append(j)
    ctx = torch.tensor(ctx).to(device)
    cnd1 = torch.tensor(cnd1).to(device)
    cnd2 = torch.tensor(cnd2).to(device)
    cnds = torch.stack([cnd1, cnd2], 1) # [bs, 2]
    cnd_ctx = torch.gather(cnds, 1, ctx.view(-1, 1)).squeeze() # bs
    for i in range(72):
        input_tensor[i, :, 2 + ctx[i]] = 1 # LOC ctx 1 0 FREQ ctx 0 1
        input_tensor[i, 20:60, 0] = [-0.4,-0.2,-0.1,0.1,0.2,0.4][cnd1[i]] + torch.randn(40) * noise_inp
        input_tensor[i, 20:60, 1] = [-0.4,-0.2,-0.1,0.1,0.2,0.4][cnd2[i]] + torch.randn(40) * noise_inp
    y = (2 * (cnd_ctx < 3).float() - 1).view(72, 1, 1).repeat(1, 61, 1)
    return input_tensor, y, cnds, ctx, cnd_ctx

def get_spike_input(bs=BATCH_SIZE, fixed_t=0):
    spike_input = torch.zeros(bs, fixed_t+40, 4).to(device)
    ctx = torch.randint(2, (bs, )).to(device)
    cnd1 = torch.randint(6, (bs, )).to(device)
    cnd2 = torch.randint(6, (bs, )).to(device)
    cnds = torch.stack([cnd1, cnd2], 1) # [bs, 2]
    cnd_ctx = torch.gather(cnds, 1, ctx.view(-1, 1)).squeeze() # bs
    mask = torch.zeros(bs, fixed_t+40, 1).to(device)
    mask[:, fixed_t+39, 0] = 1

    input_frq = torch.ones(bs, 40, device=device) * 0.8
    spike_count = torch.poisson(input_frq)
    probs = torch.tensor([39, 35, 25, 15, 5, 1]).to(device) / 40
    loc_right_prob = probs[cnd1].unsqueeze(1).repeat(1, 40)
    frq_high_prob = probs[cnd2].unsqueeze(1).repeat(1, 40)
    loc_sample = torch.distributions.binomial.Binomial(spike_count, loc_right_prob)
    loc_right = loc_sample.sample()
    loc_left = spike_count - loc_right
    frq_sample = torch.distributions.binomial.Binomial(spike_count, frq_high_prob)
    frq_high = frq_sample.sample()
    frq_low = spike_count - frq_high
    spike_input[:, fixed_t:, 0] = (loc_right - loc_left) * 0.1 # LOC input
    spike_input[:, fixed_t:, 1] = (frq_high - frq_low) * 0.1 # FREQ input

    for i in range(bs):
        spike_input[i, :, 2 + ctx[i]] = 1 # LOC ctx 1 0 FREQ ctx 0 1

    y = (2 * (cnd_ctx < 3).float() - 1).view(bs, 1, 1).repeat(1, fixed_t+40, 1)

    return spike_input, y, mask, cnds, ctx, cnd_ctx

def get_random_input(bs, ctx, fixed_t=0, hz=1, device=device):
    spike_input = torch.zeros(bs, fixed_t+40, 4).to(device)
    input_frq = torch.ones(bs, 40, device=device) * 0.02 * hz
    spike_count = torch.poisson(input_frq)
    loc_right_prob = torch.ones_like(spike_count) * 0.5
    frq_high_prob = torch.ones_like(spike_count) * 0.5
    loc_sample = torch.distributions.binomial.Binomial(spike_count, loc_right_prob)
    loc_right = loc_sample.sample()
    loc_left = spike_count - loc_right
    frq_sample = torch.distributions.binomial.Binomial(spike_count, frq_high_prob)
    frq_high = frq_sample.sample()
    frq_low = spike_count - frq_high
    spike_input[:, fixed_t:, 0] = (loc_right - loc_left) * 0.1 # LOC input
    spike_input[:, fixed_t:, 1] = (frq_high - frq_low) * 0.1 # FREQ input
    spike_input[:, :, 2 + ctx] = 1
    return spike_input

def get_standard_spike_input(fixed_t=0):
    spike_input = torch.zeros(72, fixed_t+40, 4).to(device)
    ctx = list()
    cnd1 = list()
    cnd2 = list()
    for c in range(2):
        for i in range(6):
            for j in range(6):
                ctx.append(c)
                cnd1.append(i)
                cnd2.append(j)
    ctx = torch.tensor(ctx).to(device)
    cnd1 = torch.tensor(cnd1).to(device)
    cnd2 = torch.tensor(cnd2).to(device)
    cnds = torch.stack([cnd1, cnd2], 1) # [bs, 2]
    cnd_ctx = torch.gather(cnds, 1, ctx.view(-1, 1)).squeeze() # bs
    input_frq = torch.ones(72, 40, device=device) * 0.8
    spike_count = torch.poisson(input_frq)
    probs = torch.tensor([39, 35, 25, 15, 5, 1]).to(device) / 40
    loc_right_prob = probs[cnd1].unsqueeze(1).repeat(1, 40)
    frq_high_prob = probs[cnd2].unsqueeze(1).repeat(1, 40)
    loc_sample = torch.distributions.binomial.Binomial(spike_count, loc_right_prob)
    loc_right = loc_sample.sample()
    loc_left = spike_count - loc_right
    frq_sample = torch.distributions.binomial.Binomial(spike_count, frq_high_prob)
    frq_high = frq_sample.sample()
    frq_low = spike_count - frq_high
    spike_input[:, fixed_t:, 0] = (loc_right - loc_left) * 0.1 # LOC input
    spike_input[:, fixed_t:, 1] = (frq_high - frq_low) * 0.1 # FREQ input
    for i in range(72):
        spike_input[i, :, 2 + ctx[i]] = 1
    y = (2 * (cnd_ctx < 3).float() - 1).view(72, 1, 1).repeat(1, fixed_t+40, 1)
    return spike_input, y, cnds, ctx, cnd_ctx
