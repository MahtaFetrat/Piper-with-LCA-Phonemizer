import onnxruntime as ort
from transformers import AutoTokenizer
from Parsivar.normalizer import Normalizer
import numpy as np

class GE2PE_ONNX:
    def __init__(self, onnx_path, tokenizer_path, GPU=False, dictionary=None):
        self.GPU = GPU
        self.session_options = ort.SessionOptions()

        # Configure ONNX Runtime
        providers = ['CUDAExecutionProvider'] if self.GPU else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(
            onnx_path,
            sess_options=self.session_options,
            providers=providers
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.dictionary = dictionary
        self.norma = Normalizer(pinglish_conversion_needed=True)

        # Keep all your existing rule methods
        self.is_vowel = lambda char: (char in ['a', '/', 'i', 'e', 'u', 'o'])

    def rules(self, grapheme, phoneme):
        grapheme = grapheme.replace('آ', 'ءا')
        words = grapheme.split(' ')
        prons = phoneme.replace('1', '').split(' ')
        if len(words) != len(prons):
            return phoneme
        for i in range(len(words)):
            if 'ِ' not in words[i] and 'ُ' not in words[i] and 'َ' not in words[i]:
                continue
            for j in range(len(words[i])):
                if words[i][j] == 'َ':
                    if j == len(words[i]) - 1 and prons[i][-1] != '/':
                        prons[i] = prons[i] + '/'
                    elif self.is_vowel(prons[i][j]):
                        prons[i] = prons[i][:j] + '/' + prons[i][j+1:]
                    else:
                        prons[i] = prons[i][:j] + '/' + prons[i][j:]
                if words[i][j] == 'ِ':
                    if j == len(words[i]) - 1 and prons[i][-1] != 'e':
                        prons[i] = prons[i] + 'e'
                    elif self.is_vowel(prons[i][j]):
                        prons[i] = prons[i][:j] + 'e' + prons[i][j+1:]
                    else:
                        prons[i] = prons[i][:j] + 'e' + prons[i][j:]
                if words[i][j] == 'ُ':
                    if j == len(words[i]) - 1 and prons[i][-1] != 'o':
                        prons[i] = prons[i] + 'o'
                    elif self.is_vowel(prons[i][j]):
                        prons[i] = prons[i][:j] + 'o' + prons[i][j+1:]
                    else:
                        prons[i] = prons[i][:j] + 'o' + prons[i][j:]
        return ' '.join(prons)

    def lexicon(self, grapheme, phoneme):
        words = grapheme.split(' ')
        prons = phoneme.split(' ')
        output = prons
        for i in range(len(words)):
            try:
                output[i] = self.dictionary[words[i]]
                if prons[i][-1] == '1' and output[i][-1] != 'e':
                    output[i] = output[i] + 'e1'
                elif prons[i][-1] == '1' and output[i][-1] == 'e':
                    output[i] = output[i] + 'ye1'
            except:
                pass
        return ' '.join(output)

    def generate(self, input_list, batch_size=1, use_rules=False, use_dict=False, max_length=512):
        """
        Modified to work with ONNX while keeping all original functionality
        """
        output_list = []

        # Original pre-processing
        input_list = [self.norma.normalize(text).replace('ك', 'ک') for text in input_list]
        original_input = input_list
        input_list = [text.replace('ِ', '').replace('ُ', '').replace('َ', '') for text in input_list]

        for text in input_list:
            # Tokenize encoder input
            encoder_inputs = self.tokenizer(text, return_tensors="np")

            # Initialize decoder input with start token
            decoder_input_ids = np.array([[self.tokenizer.pad_token_id]])

            # Autoregressive generation
            for _ in range(max_length):
                ort_inputs = {
                    "input_ids": encoder_inputs["input_ids"],
                    "attention_mask": encoder_inputs["attention_mask"],
                    "decoder_input_ids": decoder_input_ids
                }

                outputs = self.ort_session.run(None, ort_inputs)
                next_token = np.argmax(outputs[0][:, -1, :], axis=-1)

                if next_token == self.tokenizer.eos_token_id:
                    break

                decoder_input_ids = np.concatenate(
                    [decoder_input_ids, next_token.reshape(1, 1)],
                    axis=-1
                )

            decoded = self.tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
            output_list.append(decoded)

        # Apply post-processing rules (same as original)
        if use_dict:
            for i in range(len(input_list)):
                output_list[i] = self.lexicon(input_list[i], output_list[i])

        if use_rules:
            for i in range(len(input_list)):
                output_list[i] = self.rules(original_input[i], output_list[i])

        return [i.strip() for i in output_list]
