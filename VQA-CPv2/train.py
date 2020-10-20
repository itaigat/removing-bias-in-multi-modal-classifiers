import os
import time
import torch
import utils
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from regularization import Regularization, Perturbation

hooks = {}


def ques_out_hook(module, inp, out):
    hooks['q_net'] = inp
    hooks['q_net_out'] = out


def v_out_hook(module, inp, out):
    hooks['v_net'] = inp
    hooks['v_net_out'] = out


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output, eval_each_epoch, inf):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    all_results = []
    exp_name = f'{inf.lambda_}_{datetime.now().hour}_{datetime.now().minute}'
    writer = SummaryWriter(f'debug/{exp_name}')

    total_step = 0

    model.q_net.register_forward_hook(ques_out_hook)
    model.v_net.register_forward_hook(v_out_hook)

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()

        for i, (v, q, a, b) in tqdm(enumerate(train_loader), ncols=100,
                                    desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            total_step += 1
            v = v.cuda()
            q = q.cuda()
            a = a.cuda()
            b = b.cuda()

            pred, loss = model(v, None, q, a, b)

            if (loss != loss).any():
                raise ValueError("NaN loss")

            if inf.lambda_ > 0:
                with torch.backends.cudnn.flags(enabled=False):
                    inf_pred = Perturbation.get_expanded_logits(pred, inf.n_samples)

                    a_inf = Perturbation.perturb_tensor(a, inf.n_samples, perturbation=False)
                    b_inf = Perturbation.perturb_tensor(b, inf.n_samples, perturbation=False)

                    inf_logits, _ = model(v, None, q, a_inf, b_inf, inf=True)

                    influence_loss = nn.functional.binary_cross_entropy_with_logits(inf_pred, inf_logits)
                    gradients = torch.autograd.grad(influence_loss, [hooks['q_net'][0], hooks['v_net'][0]],
                                                    create_graph=True)

                    grads = [Regularization.get_batch_norm(grad=gradient, loss=influence_loss) for gradient in gradients]

                    inf_scores = torch.stack(grads)
                    reg = Regularization.get_regularization_term(inf_scores, inf.norm, inf.optim_method)
                    loss += inf.lambda_ * reg

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        run_eval = eval_each_epoch or (epoch == num_epochs - 1)

        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader)
            results["epoch"] = epoch + 1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score
            all_results.append(results)

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]

        logger.write('epoch %d, time: %.2f' % (epoch + 1, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        writer.add_scalar('Accuracy/Train', train_score, epoch)

        if run_eval:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            writer.add_scalar('Accuracy/Eval', 100 * eval_score, epoch)

        if 100 * eval_score >= 54:
            model_path = os.path.join(output, f'model_{eval_score}.pth')
            torch.save(model.state_dict(), model_path)


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0

    all_logits = []
    all_bias = []
    for v, q, a, b in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred, _ = model(v, None, q, None, None)
        all_logits.append(pred.data.cpu().numpy())

        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)
        all_bias.append(b)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    results = dict(
        score=score,
        upper_bound=upper_bound,
    )
    return results
