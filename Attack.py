import torch
import torch.optim as optim
from FastText import FastText,get_model
from main import predict

from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

device = torch.device("cuda:1")

def insertion(origin_tensor, insertion_tensor, text_length , insert_pos=0):
    target_tensor = origin_tensor.detach().numpy()
    target_tensor = torch.from_numpy(target_tensor)

    t0, t1 = target_tensor.split((insert_pos, text_length-insert_pos),dim=1)
    target_tensor = torch.cat((t0,insertion_tensor),dim=1)
    target_tensor = torch.cat((target_tensor,t1),dim=1)

    target_tensor = target_tensor.to(device)

    return target_tensor

def attack(origin_tensor, model, origin_label, target_label):
    max_iterations = 10000
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    model = model.to(device)

    text_length = list(origin_tensor.size())[1]
    for insert_pos in range(text_length+1):
        insertion_tensor = torch.rand(1, 1, 300)
        insertion_tensor.requires_grad = True
        optimizer = optim.Adam([insertion_tensor], lr=0.01)
        optimizer.zero_grad()

        for iteration in range(1, max_iterations + 1):
            target_tensor = insertion(origin_tensor, insertion_tensor, text_length, insert_pos)
            output = model(target_tensor)
            # if iteration == 1:
            #     print("first prediction:{}".format(output))

            predict_label = output.argmax(1).item() + 1
            if predict_label == target_label:
                print("attack success,insert_pos:{},iteration:{},output:{}".format(insert_pos, iteration,list(output)))
                break

            f = output[0][origin_label-1] - output[0][target_label-1]
            loss = f
            loss.backward()
            optimizer.step()

            # if iteration%100 == 0:
            #     print("attack iteration:{},output:{}".format(iteration,output))

def ToEmbed(text, model):
    tokenizer = get_tokenizer("basic_english")
    tokenized_text = tokenizer(text)

    origin_tensor = torch.tensor([model.vocab[token]
                                  for token in ngrams_iterator(tokenized_text, 1)])
    origin_tensor = torch.stack([origin_tensor], 0)
    origin_tensor = model.embed(origin_tensor)

    return origin_tensor

if __name__ == "__main__":
    model, _, _ = get_model(True)
    model.eval()

    text = "WASHINGTON, Aug. 19 (Xinhuanet) -- Andre Agassi cruised into quarter-finals in Washington Open tennis with a 6-4, 6-2 victory over Kristian Pless of Denmark here on Thursday night."

    orig_label = 2
    target_label = 4

    origin_tensor = ToEmbed(text, model)

    # text_tensor = tokenize(ex_text_str, sentence_max_size, vocab)
    # predict(text, model, model.vocab)

    attack(origin_tensor, model, orig_label, target_label)


