import torch

from model import CascadeRCNN
from config import cfg
from dataset import WSGISDDataset, cls2idx

ds = WSGISDDataset(root='datasets/wgisd', resize=(800, 1280))
dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ds.collate_fn)
cfg.NUM_CLASSES = len(cls2idx)

model = CascadeRCNN(cfg)
print(model)
model = model.cuda()

if 0:
    from tqdm import tqdm
    for _ in tqdm(dl):
        pass

if 0:
    batched_inputs = next(iter(dl))
    result = model(batched_inputs)
    for k, v in result.items():
        print(k, v.item())

    model.eval()
    result = model(batched_inputs)
    import pdb; pdb.set_trace()

#
# epochs = 20
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# for epoch in range(epochs):
#     for step, batched_inputs in enumerate(dl):
#         optimizer.zero_grad()
#         result = model(batched_inputs)
#
#         losses = []
#         print(f'\nEPOCH: {epoch}    STEP: {step}')
#         for k, v in result.items():
#             losses.append(v)
#             print(k, v.item())
#
#         loss = sum(losses)
#         print(f'total loss: {loss.item()}\n')
#         loss.backward()
#         optimizer.step()
