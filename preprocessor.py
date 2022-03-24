import numpy as np
import itertools
import torch
from torch.nn.utils.rnn import pad_sequence
from poppler import load_from_file
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import Dataset
import pandas as pd
import glob
import json
import re


class Preprocessor:
    @staticmethod
    def parse_as_dialogue(text: str) -> [(str, str)]:
        """
        Text from PDFs will have various annotations (e.g. page numbers, summary, etc.) that are useless to our model.
        This function should remove them and return only the dialogue from a given document.

        :param text: Some string from the PDFs consisting of dialogue and various annotations.
        :return: An array of tuples in which the first element gives the speaker name and the second gives their dialogue,
                 with these tuples listed in the same order as they appeared in the input string.
        """
        # Remove annotations at top of page and embedded clips.
        no_page_annots = '\n'.join(filter(
            lambda x: not (
                    '1st Ear' in x or  # Page header line 1.
                    'DRAFT' in x or  # Page header line 2.
                    '\u000c' in x or  # Form feed which is frequently on its own line.
                    re.match(r' +[0-9]+', x) or  # Page number (occasionally on its own line).
                    '_Clip' in x  # Embedded audio clip.
            ), text.split('\n')))

        # List of all participants within the training PDFs.
        participants = ['Connally', 'Rogers', 'Ziegler', 'Nixon', 'Kendall', 'Westmoreland', 'Garment', 'Haig',
                        'Kissinger', 'Ehrlichman', 'Helms', 'Finch', 'Haldeman', 'Hademan', 'Humphrey', 'Laird',
                        'Flanigan', 'Heath', 'Bluhdorn', 'Butterfield', 'President', 'Echeverria', 'Colson']

        # Dialogue regex should be:
        #   (1) name of some participant (capturing), followed by;
        #   (2) colon and space (non-capturing), followed by;
        #   (3) some line of dialogue (capturing), which may contain newlines, but may not contain the name of any
        #       speaker followed by a colon (as this indicates a new line of dialogue has begun).
        dialogue_regex = f'({"|".join([f"{x}" for x in participants])})' \
                         f'(?:: )' \
                         f'((?:(?!{"|".join([f"{x}:" for x in participants])}).+\n)+)'
        dialogue_lines = [(s, re.sub(r'[\[\]\n]|\[Unclear]', '', l))
                          for (s, l) in re.findall(dialogue_regex, no_page_annots)]
        dialogue_lines = [(s, re.sub(r'\u2019', '\'', l).lower()) for (s, l) in dialogue_lines]
        # There is one particular PDF where Nixon is labeled as "President" in his dialogue. To avoid confusion, ensure
        # that "Nixon" is the only speaker ID used to identify Nixon.
        return [(s if s != 'President' else 'Nixon', d) for (s, d) in dialogue_lines]

    @staticmethod
    def read_pdf(path: str) -> [str]:
        """
        Transcribe a PDF page-by-page into a list of strings.

        :param path: The path to the pdf which will be read.
        :return: A list in which element n gives the transcribed text of page n of the given pdf.
        """
        doc = load_from_file(path)
        pages = []
        page_num = 0

        while (pg := doc.create_page(page_num)):
            try:
                pages.append(pg.text())
                page_num += 1
            # invalid page index.
            except AttributeError:
                break

        return pages


@dataclass
class ConversationData(Dataset):
    # Set of named speakers involved in the conversation.
    speakers: {str}
    # A list of conversations, in which each conversation consists of a list of tuples, where each tuple gives the
    # speaker of a line of dialogue and the line itself. These lines should be ordered based on how they appeared within
    # the document.
    conversations: [[(str, str)]]
    tokenized_convos: [int]
    contexts: pd.DataFrame

    def __init__(self, *, json_path=None, input_dir=None, cutoff=None):
        """
        Constructor for ConversationData (kw-only). Should be called with either JSON path or input_dir, and not both.

        :param json_path: A path to a JSON file where a ConversationData instance has previously been serialized.
        :param input_dir: A path to a directory containing PDFs from which we can derive ConversationData.
        """
        self.tokenizer = AutoTokenizer.from_pretrained('DialoGPT-small')

        if json_path is not None and input_dir is not None:
            print('[WARNING] ConversationData constructor should be called with either JSON path or input_dir. '
                  'Defaulting to JSON path.')

        if json_path is not None:
            with open(json_path, 'r') as infile:
                data = json.load(infile)
            self.speakers = set(data['speakers'])

            # Annoyingly, JSON doesn't have tuples, so we're forced to serialize tuples as 2-element lists.
            # When parsing JSON, convert back to tuples.
            self.conversations = [[(s, d) for s, d in conversation] for conversation in data['conversations']]
            if cutoff == None:
                self.tokenized_convos = data['tokenized_convos']
            else:
                self.tokenized_convos = data['tokenized_convos'][:cutoff]

        elif input_dir is not None:
            self.speakers = set()
            self.conversations = []
            self.tokenized_convos = []
            for pdf in glob.glob(f'{input_dir}/*.pdf'):
                text = '\n'.join(Preprocessor.read_pdf(pdf))
                conversation = Preprocessor.parse_as_dialogue(text)
                self.speakers |= set([speaker for (speaker, _) in conversation])
                self.conversations.append(conversation)
                self.contexts = self.generate_contexts()
                # tokenized_convos *should* be 2d.
                for _, row in self.contexts.iterrows():
                    self.tokenized_convos += self.generate_conv(row)

        else:
            raise ValueError("Constructor should be called with either JSON path or input dir.")

    def parse_file(self, path: str) -> None:
        """
        Given a path to some PDF containing the nixon tapes, parse it and update the speakers/conversations accordingly.

        :param path: The path to the PDF to parse.
        :return: Nothing. (Also, appreciate for a moment how stupid python type annotations are. Returning "None" has
                 the same type annotation as returning nothing at all, because null is a useless type. Blame Tony Hoare,
                 I suppose.)
        """
        text = '\n'.join(Preprocessor.read_pdf(path))
        dialogue = Preprocessor.parse_as_dialogue(text)
        self.speakers |= [speaker for (speaker, _) in dialogue]
        self.conversations.append(dialogue)

    def save_to_file(self, output_path: str) -> None:
        """
        Save the conversation data to a JSON file.

        :param output_path: The path of the JSON file to which this data will be saved.
        :return: Nothing.
        """
        serialized_data = {
            'speakers': list(self.speakers),
            'conversations': self.conversations,
            'tokenized_convos': self.tokenized_convos
        }

        with open(output_path, 'w+') as outfile:
            json.dump(serialized_data, outfile)

    def generate_contexts(self) -> pd.DataFrame:
        history_len = 7
        contexts = []
        lines = [[line for (_, line) in convo] for convo in self.conversations]
        for conversation in lines:
            contexts += list(itertools.chain([conversation[i:(i-1-history_len):-1]
                                         for i in range(history_len, len(conversation))]))

        columns = ['response', 'context'] + [f'context/{i}' for i in range(history_len - 1)]
        print(f'Len is {len(contexts)}')
        # Contexts is 2d arr, as intended.
        df = pd.DataFrame.from_records(contexts, columns=columns)
        df.to_csv('train_df.csv')
        return df

    def generate_conv(self, row):
        # EOS token signals that a speaker has finished speaking. GPT uses this.
        conv = list(reversed([self.tokenizer.encode(word) + [self.tokenizer.eos_token_id] for word in row
                              if word is not None]))
        # returns 1d array.
        return list(itertools.chain(conv))

    def save_to_csv(self):
        arr = []
        cols = [
            "@id",
            "userProfile.userID.$",
            "sms_text",
            "userProfile.country.$",
            "userProfile.age.$",
            "userProfile.gender.$",
            "srcNumber.$",
            "phoneModel.@manufactuer",
            "phoneModel.@smartphone",
            "userProfile.frequency.$",
        ]
        lines = 0
        break_out = False
        for convo in self.conversations:
            for speaker, line in convo:
                arr += [['', '', line, 'SG', '', '', '', '', '', '']]
                lines += 1


        df = pd.DataFrame.from_records(arr, columns=cols)
        df.to_csv('train_df.csv')



    def __len__(self):
        return len(self.tokenized_convos)

    def __getitem__(self, ind):
        # You may need to pad data before this?
        return torch.tensor(self.tokenized_convos[ind], dtype=torch.long)


if __name__ == "__main__":
    data = pd.read_csv('train_df.csv')
    print(len(data))
    data = ConversationData(input_dir="./training")
    print(data.contexts.head())
    print(len(data.contexts))
    # data.save_to_csv()
    # data.save_to_file('./training/training.json')
