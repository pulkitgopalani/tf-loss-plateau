import random
import numpy as np
import torch

class MovingWindowSum:
    def __init__(self, min_num=1, max_num=16, k=2, p=17, sep=17, device="cuda"):
        self.min_num = min_num
        self.max_num = max_num
        self.k = k
        self.p = p
        self.sep = sep
        self.device = device
        assert self.p > self.max_num

    @torch.no_grad()
    def sample(
        self,
        num_samples,
        num_tokens,
    ):
        random_ints = torch.randint(
            low=self.min_num, high=self.max_num + 1, size=(num_samples, num_tokens)
        ).to(self.device)

        random_ints_np = random_ints.detach().cpu().numpy()
        convolution = torch.stack(
            [
                torch.from_numpy(
                    np.convolve(
                        random_ints_np[i],
                        np.ones(self.k),
                        mode="valid",
                    )
                )
                for i in range(random_ints.shape[0])
            ]
        )

        moving_sum = random_ints.clone().detach()
        moving_sum[:, self.k - 1 :] = convolution

        # for i in range(num_samples):
        #     for j in range(0, self.k - 1):
        #         if moving_sum[i, j] != random_ints[i, j]:
        #             print(f"ERROR! {i} {j}")
        #     for j in range(self.k - 1, num_tokens):
        #         if moving_sum[i, j] != torch.sum(random_ints[i, j-self.k+1:j+1]):
        #             print(f"ERROR! {i} {j}")

        # exit()
        samples = (
            torch.cat(
                [
                    random_ints,
                    self.sep * torch.ones(size=(num_samples, 1)).to(self.device),
                    torch.remainder(input=moving_sum, other=self.p),
                ],
                axis=-1,
            )
            .to(int)
            .detach()
        )

        return samples

class ConcatMovingWindowSum:
    def __init__(
        self,
        min_num=1,
        max_num=16,
        k=2,
        p=17,
        mix_type="123_disjoint_454",
        sep=17,
        device="cuda",
    ):
        self.min_num = min_num
        self.max_num = max_num
        self.k = k
        self.p = p
        self.mix_type = mix_type
        self.sep = sep
        self.device = device
        assert self.p > self.max_num

    @torch.no_grad()
    def sample(
        self,
        num_samples,
        num_tokens,
    ):
        random_ints = torch.randint(
            low=self.min_num, high=self.max_num + 1, size=(num_samples, num_tokens)
        ).to(self.device)

        random_ints_np = random_ints.detach().cpu().numpy()
        convolution2 = torch.stack(
            [
                torch.from_numpy(
                    np.convolve(
                        random_ints_np[i],
                        np.ones(2),
                        mode="valid",
                    )
                )
                for i in range(random_ints.shape[0])
            ]
        )

        if self.mix_type == "124_stair_653":
            convolution4 = torch.stack(
                [
                    torch.from_numpy(
                        np.convolve(
                            random_ints_np[i],
                            np.ones(4),
                            mode="valid",
                        )
                    )
                    for i in range(random_ints.shape[0])
                ]
            )

            # Stair 124 - 653
            moving_sum = random_ints[:, :14].clone().detach()
            moving_sum[:, 6:11] = convolution2[:, :5]
            moving_sum[:, 11:14] = convolution4[:, :3]

        elif self.mix_type == "124_disjoint_444":
            convolution4 = torch.stack(
                [
                    torch.from_numpy(
                        np.convolve(
                            random_ints_np[i],
                            np.ones(4),
                            mode="valid",
                        )
                    )
                    for i in range(random_ints.shape[0])
                ]
            )

            # Disjoint 124 - 444
            moving_sum = random_ints[:, :12].clone().detach()
            moving_sum[:, 4:8] = convolution2[:, 4:8]
            moving_sum[:, 8:12] = convolution4[:, 9:]

        elif self.mix_type == "124_disjoint_453":
            convolution4 = torch.stack(
                [
                    torch.from_numpy(
                        np.convolve(
                            random_ints_np[i],
                            np.ones(4),
                            mode="valid",
                        )
                    )
                    for i in range(random_ints.shape[0])
                ]
            )

            # Disjoint 124 - 453
            moving_sum = random_ints[:, :12].clone().detach()
            moving_sum[:, 4:9] = convolution2[:, 4:9]
            moving_sum[:, 9:12] = convolution4[:, 10:]

        elif self.mix_type == "123_disjoint_454":
            convolution3 = torch.stack(
                [
                    torch.from_numpy(
                        np.convolve(
                            random_ints_np[i],
                            np.ones(3),
                            mode="valid",
                        )
                    )
                    for i in range(random_ints.shape[0])
                ]
            )

            # Disjoint 123 - 454
            # print(convolution3.shape)
            # exit()
            moving_sum = random_ints[:, :13].clone().detach()
            moving_sum[:, 4:9] = convolution2[:, 4:9]
            moving_sum[:, 9:13] = convolution3[:, 10:]

        else:
            raise ValueError("Data mixture type not defined")

        samples = (
            torch.cat(
                [
                    random_ints,
                    self.sep * torch.ones(size=(num_samples, 1)).to(self.device),
                    torch.remainder(input=moving_sum, other=self.p),
                ],
                axis=-1,
            )
            .to(int)
            .detach()
        )

        return samples


class PrefixSum:
    def __init__(self, min_num=1, max_num=16, p=17, no_repeat=False, device="cuda"):
        self.min_num = min_num
        self.max_num = max_num
        self.p = p
        self.no_repeat = no_repeat
        assert self.p > self.max_num

        self.device = device

    @torch.no_grad()
    def sample(self, num_samples, num_tokens):
        
        if self.no_repeat:
            random_ints = torch.arange(start=self.min_num, end=self.max_num+1).view(-1, num_tokens).repeat(num_samples, 1).to(self.device)

            for i in range(num_samples):
                random_ints[i, :] = random_ints[i, torch.randperm(num_tokens)]

        else:
            random_ints = torch.randint(
                low=self.min_num, high=self.max_num + 1, size=(num_samples, num_tokens)
            ).to(self.device)

        prefix_mod_sum = torch.remainder(torch.cumsum(random_ints, dim=-1), self.p)

        samples = (
            torch.cat(
                [
                    random_ints,
                    self.p * torch.ones(size=(num_samples, 1)).to(self.device),
                    prefix_mod_sum,
                ],
                axis=-1,
            )
            .to(int)
            .detach()
        )

        return samples


class MultiDigitSum:
    def __init__(
        self, n=5, reverse=False, plus_token=10, equal_token=11, device="cuda"
    ):
        self.n = n
        self.reverse = reverse
        self.plus_token = plus_token
        self.equal_token = equal_token
        self.device = device

        self.vocab = {}
        for x in range(10):
            self.vocab.update({f"{x}": x})
        self.vocab.update({"+": self.plus_token})
        self.vocab.update({"=": self.equal_token})

        self.tokenize = lambda x: self.vocab[x]

    @torch.no_grad()
    def sample(self, num_samples):

        samples = torch.zeros((num_samples, 3 * self.n + 3), dtype=int).to(self.device)

        for idx in range(num_samples):
            a = random.randint(1, 10**self.n - 1)
            b = random.randint(1, 10**self.n - 1)

            if self.reverse:
                list_sample = list(
                    map(
                        self.tokenize,
                        list(
                            f"{str(a).zfill(self.n)}+{str(b).zfill(self.n)}={str(a+b).zfill(self.n+1)[::-1]}"
                        ),
                    )
                )
            else:
                list_sample = list(
                    map(
                        self.tokenize,
                        list(
                            f"{str(a).zfill(self.n)}+{str(b).zfill(self.n)}={str(a+b).zfill(self.n+1)}"
                        ),
                    )
                )

            samples[idx] = torch.tensor(list_sample)

        return samples


class Histogram:
    def __init__(self, alphabet_size=26, device="cuda"):
        self.alphabet_size = alphabet_size
        self.device = device

    @torch.no_grad()
    def count(self, sample):
        counts = torch.cat(
            [
                sample,
                torch.zeros((1,), dtype=int).to(self.device),
                torch.bincount(sample)[sample],
            ],
            axis=-1,
        )

        return counts

    @torch.no_grad()
    def sample(self, num_samples, num_tokens):
        
        random_array = torch.randint(
            low=1, high=self.alphabet_size + 1, size=(num_samples, num_tokens)
        ).to(self.device)

        samples = torch.empty(num_samples, 2 * num_tokens + 1).to(self.device)
        for i in range(num_samples):
            samples[i] = self.count(random_array[i])

        data_samples = samples.to(int).detach()
        return data_samples



class Permute:
    def __init__(self, min_num, max_num, sep=0, compose=1, max_len=None, device="cuda"):
        self.min_num = min_num
        self.max_num = max_num
        
        self.sep = sep
        self.compose = compose
        self.max_len = max_len
        
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        num_samples,
        num_tokens,
    ):

        random_ints = torch.randint(
            low=self.min_num, high=self.max_num + 1, size=(num_samples, num_tokens)
        ).to(self.device)

        pi = torch.stack(
            [torch.randperm(num_tokens) for _ in range(num_samples)]
        ).to(self.device)

        if self.compose == 1:
            new_pi = pi.clone().detach()

        elif self.compose == 2:
            new_pi = torch.stack([pi[i, pi[i]] for i in range(num_samples)], dim=0)

        elif self.compose == 3:
            new_pi_1 = torch.stack([pi[i, pi[i]] for i in range(num_samples)], dim=0)
            new_pi = torch.stack(
                [new_pi_1[i, pi[i]] for i in range(num_samples)], dim=0
            )

        elif self.compose == 4:
            new_pi_1 = torch.stack([pi[i, pi[i]] for i in range(num_samples)], dim=0)
            new_pi_2 = torch.stack(
                [new_pi_1[i, pi[i]] for i in range(num_samples)], dim=0
            )
            new_pi = torch.stack(
                [new_pi_2[i, pi[i]] for i in range(num_samples)], dim=0
            )

        permuted_ints = torch.stack(
            [random_ints[i, new_pi[i]] for i in range(num_samples)],
            dim=0,
        )

        samples = (
            torch.cat(
                [
                    random_ints[:num_samples],
                    torch.zeros(size=(num_samples, 1)).to(self.device),
                    pi[:num_samples] + 1,
                    self.sep * torch.ones(size=(num_samples, 1)).to(self.device),
                    permuted_ints[:num_samples],
                ],
                axis=-1,
            )
            .to(int)
            .detach()
        )

        return samples 


class Copy:
    def __init__(self, min_num=1, max_num=16, device="cuda"):
        self.min_num = min_num
        self.max_num = max_num
        self.device = device

    @torch.no_grad()
    def sample(self, num_samples, num_tokens):
        random_ints = torch.randint(
            low=self.min_num, high=self.max_num + 1, size=(num_samples, num_tokens)
        ).to(self.device)

        samples = (
            torch.cat(
                [
                    random_ints,
                    torch.zeros(size=(num_samples, 1)).to(self.device),
                    random_ints,
                ],
                axis=-1,
            )
            .to(int)
            .detach()
        )

        return samples

class Reverse:
    def __init__(self, min_num=1, max_num=16, device="cuda"):
        self.min_num = min_num
        self.max_num = max_num
        self.device = device

    @torch.no_grad()
    def sample(self, num_samples, num_tokens):
        random_ints = torch.randint(
            low=self.min_num, high=self.max_num + 1, size=(num_samples, num_tokens)
        ).to(self.device)

        reversed = torch.flip(random_ints, dims=(-1,))

        samples = (
            torch.cat(
                [
                    random_ints,
                    torch.zeros(size=(num_samples, 1)).to(self.device),
                    reversed,
                ],
                axis=-1,
            )
            .to(int)
            .detach()
        )

        return samples


class Repeat:
    def __init__(self, min_num=1, max_num=16, k=1, pos=0, device="cuda"):
        self.min_num = min_num
        self.max_num = max_num
        self.p = max_num + 1

        self.sep = max_num + 1
        self.pos = pos
        self.k = k

        self.device = device

    @torch.no_grad()
    def sample(self, num_samples, num_tokens):
        input_random_ints = torch.randint(
            low=self.min_num, high=self.max_num + 1, size=(num_samples, num_tokens)
        ).to(self.device)
        output_random_ints = torch.empty_like(input_random_ints)

        if self.k == 1:
            output_random_ints = input_random_ints[:, self.pos].view(-1, 1).repeat(1, num_tokens)

        elif self.k == 2:
            output_random_ints[:, :num_tokens // 2] = input_random_ints[:, self.pos].view(-1, 1).repeat((1, num_tokens//2))
            output_random_ints[:, num_tokens // 2:] = torch.remainder(output_random_ints[:, :num_tokens // 2] + 1, self.p)
            
        elif self.k == 4:
            output_random_ints[:, :num_tokens // 4] = input_random_ints[:, self.pos].view(-1, 1).repeat((1, num_tokens // 4))
            output_random_ints[:, num_tokens // 4 : num_tokens // 2] = torch.remainder(output_random_ints[:, :num_tokens // 4] + 1, self.p)
            output_random_ints[:, num_tokens // 2: (3*num_tokens) // 4] = torch.remainder(output_random_ints[:, :num_tokens // 4] + 2, self.p)
            output_random_ints[:, (3*num_tokens) // 4: ] = torch.remainder(output_random_ints[:, :num_tokens // 4] + 3, self.p)

        samples = (
            torch.cat(
                [
                    input_random_ints,
                    self.sep * torch.ones(size=(num_samples, 1)).to(self.device),
                    output_random_ints,
                ],
                axis=-1,
            )
            .to(int)
            .detach()
        )

        return samples

if __name__ == "__main__":
    pass