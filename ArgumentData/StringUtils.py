import math
import random

import nltk
import torch
import bert_score

from typing import Optional, Tuple, Literal, Any
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from loguru import logger

from nltk import sent_tokenize, word_tokenize, pos_tag_sents
from nltk.corpus import wordnet

import spacy
import collections
from pattern.text.en import tenses as f_tenses, conjugate as f_conjugate

paraphrase_model: Optional[Tuple[PegasusTokenizer, PegasusForConditionalGeneration]] = None
summarization_model: Optional[Tuple[PegasusTokenizer, PegasusForConditionalGeneration]] = None
bertscore_model: Optional[bert_score.BERTScorer] = None
spacy_nlp: Optional[Any] = None


# noinspection PyBroadException
def paraphrase(text: str, temperature: Optional[float] = None,
               paraphrase_model_name: str = "tuner007/pegasus_paraphrase", avoid_equal_return: bool = True,
               maximize_dissimilarity: bool = False, fast: bool = False) -> str:
    global paraphrase_model
    global bertscore_model
    if paraphrase_model is None:
        logger.warning("The paraphrase model is not loaded until yet - let's do this (target: {})",
                       paraphrase_model_name)
        try:
            paraphrase_model = (
                PegasusTokenizer.from_pretrained(paraphrase_model_name),
                PegasusForConditionalGeneration.from_pretrained(paraphrase_model_name).to(
                    "cuda" if torch.cuda.is_available() else "cpu")
            )
            logger.success("Loaded the paraphrase-model: {}",
                           " + ".join(map(lambda p: str(p) if len(str(p)) < 20 else "{}...".format(str(p)[:15]),
                                          paraphrase_model)))
        except Exception:
            logger.opt(exception=True).critical("Something went wrong - paraphrasing disabled!")
            return text

    input_encoded = paraphrase_model[0](text=[text],
                                        padding=not fast,
                                        truncation=True,
                                        max_length=paraphrase_model[0].model_max_length,
                                        is_split_into_words=False,
                                        return_tensors="pt",
                                        return_token_type_ids=False,
                                        return_length=True,
                                        return_overflowing_tokens=True,
                                        verbose=not fast)

    length_input: int = torch.squeeze(input_encoded.pop("length")).item()
    overflow: int = torch.squeeze(input_encoded.pop("num_truncated_tokens")).item()
    overflow_str: str = paraphrase_model[0].decode(token_ids=torch.squeeze(input_encoded.pop("overflowing_tokens")),
                                                   skip_special_tokens=False, clean_up_tokenization_spaces=True)
    logger.debug("Your encoded input has the length of {} tokens (beware the model's max-length of {})",
                 length_input+max(0, overflow), paraphrase_model[0].model_max_length)
    if overflow > 0:
        logger.info("Your input was truncated ({} tokens: \"{}\")", overflow, overflow_str)
    if torch.cuda.is_available():
        logger.trace("The input-tensor ({}) must be first shifted to the GPU!", input_encoded)
        input_encoded = input_encoded.to(paraphrase_model[1].device)

    paraphrases_original = paraphrase_model[0].batch_decode(
        paraphrase_model[1].generate(**input_encoded,
                                     max_length=30 if fast else min(paraphrase_model[0].model_max_length,
                                                                    math.ceil(length_input*1.2)),
                                     num_beams=2+5*int(not fast)+int(avoid_equal_return)+3*int(maximize_dissimilarity),
                                     num_return_sequences=1+2*int(avoid_equal_return)+3*int(maximize_dissimilarity),
                                     temperature=temperature or (1.1+.5*int(maximize_dissimilarity))),
        skip_special_tokens=True, clean_up_tokenization_spaces=True)

    logger.trace("Received following texts: {}", paraphrases_original)

    paraphrases = paraphrases_original.copy()
    if avoid_equal_return:
        paraphrases = [p for p in paraphrases_original if p.lower() != text.lower()]
        if len(paraphrases) == 0:
            logger.warning("You want to avoid equal returns, but we received only equal returns ({} times). "
                           "Maybe you should think about disabling the fast mode?", len(paraphrases_original))
            return paraphrases_original[0]
        elif len(paraphrases) == 1:
            logger.trace("There is no choice...")
            return paraphrases[0]

    if maximize_dissimilarity:
        if bertscore_model is None:
            logger.info("We have to load the BERTscorer-model first!")
            bertscore_model = bert_score.BERTScorer(idf=False, rescale_with_baseline=False, use_fast_tokenizer=fast,
                                                    device="cuda" if torch.cuda.is_available() else "cpu", lang="en")
            logger.debug("Initializes the BERTscore: {}", bertscore_model)

        paraphrases = [
            (
                p, torch.sum(
                    bertscore_model.score(cands=[p]*len(paraphrases), refs=paraphrases, verbose=not fast,
                                          batch_size=len(paraphrases))[-1]
                ).item()
             )
            for p in paraphrases
        ]
        logger.trace("Calculated for each paraphrase the summed F1-BERTscores to all other paraphrases "
                     "(including itself): {}", paraphrases)
        paraphrases.sort(key=lambda t: t[-1], reverse=False)
        logger.debug("Sorted the paraphrases - the most dissimilar paraphrase is: {}", paraphrases[0])
        paraphrases = [p for p, s in paraphrases]
    elif not fast:
        random.shuffle(paraphrases)
        logger.trace("Shuffled the paraphrases: {} -> {}", paraphrases_original, paraphrases)

    logger.debug("The final paraphrase to \"{}\" is \"{}\"", text, paraphrases[0])

    return paraphrases[0]


# noinspection PyBroadException
def summarize(text: str, text_pair: Optional[str] = None, summarization_model_name: str = "google/pegasus-xsum",
              temperature: Optional[float] = None, fast: bool = False) -> str:
    logger.trace("WARNING! Summarizing argumentative text can conclude... into a conclusion or some strange added/ "
                 "extracted facts. It's rather unlikely the the original meaning is 100% preserved. Experimental!")
    logger.trace("The \"Mixed & Stochastic\" model has the following changes (from pegasus-large in the paper): "
                 "trained on both C4 and HugeNews (dataset mixture is weighted by their number of examples). "
                 "trained for 1.5M instead of 500k (we observe slower convergence on pretraining perplexity). "
                 "the model uniformly sample a gap sentence ratio between 15% and 45%. importance sentences are "
                 "sampled using a 20% uniform noise to importance scores. the sentencepiece tokenizer is updated to "
                 "be able to encode newline character.")

    global summarization_model

    if summarization_model is None:
        logger.warning("The paraphrase model is not loaded until yet - let's do this (target: {})",
                       summarization_model_name)
        try:
            summarization_model = (
                PegasusTokenizer.from_pretrained(summarization_model_name),
                PegasusForConditionalGeneration.from_pretrained(summarization_model_name).to(
                    "cuda" if torch.cuda.is_available() else "cpu")
            )
            logger.success("Loaded the summarization-model: {}",
                           " + ".join(map(lambda p: str(p) if len(str(p)) < 20 else "{}...".format(str(p)[:15]),
                                          summarization_model)))
        except Exception:
            logger.opt(exception=True).critical("Something went wrong - summarization disabled!")
            return text_pair or text

    input_encoded = summarization_model[0](text=[text],
                                           text_pair=None if text_pair is None else [text_pair],
                                           padding=not fast,
                                           truncation=True,
                                           max_length=summarization_model[0].model_max_length,
                                           is_split_into_words=False,
                                           return_tensors="pt",
                                           return_token_type_ids=False,
                                           return_length=True,
                                           return_overflowing_tokens=True,
                                           verbose=not fast)

    length_input: int = torch.squeeze(input_encoded.pop("length")).item()
    overflow: int = torch.squeeze(input_encoded.pop("num_truncated_tokens")).item()
    overflow_str: str = summarization_model[0].decode(token_ids=torch.squeeze(input_encoded.pop("overflowing_tokens")),
                                                      skip_special_tokens=False, clean_up_tokenization_spaces=True)
    logger.debug("Your encoded input has the length of {} tokens (beware the model's max-length of {})",
                 length_input + max(0, overflow), summarization_model[0].model_max_length)
    if overflow > 0:
        logger.info("Your input was truncated ({} tokens: \"{}\")", overflow, overflow_str)
    if torch.cuda.is_available():
        logger.trace("The input-tensor ({}) must be first shifted to the GPU!", input_encoded)
        input_encoded = input_encoded.to(summarization_model[1].device)

    summarization = summarization_model[0].batch_decode(
        summarization_model[1].generate(**input_encoded,
                                        max_length=15 if fast else min(summarization_model[0].model_max_length,
                                                                       max(math.ceil(length_input * .15), 10)),
                                        num_beams=2 if fast else 5,
                                        num_return_sequences=1,
                                        temperature=temperature),
        skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    logger.debug("Fetched following summarization: \"{}\"", summarization)

    return summarization


def add_prefix(text: str, part: Literal["undefined", "premise", "conclusion"] = "undefined") -> str:
    logger.trace("Enriching \"{}\" ({})", text, part)

    if part == "premise":
        return "{} {}{}".format(
            random.choice(["In my opinion,", "Let's assume:", "Nevertheless,"]),
            text[0].lower(),
            text[1:]
        )
    elif part == "conclusion":
        return "{} {}{}".format(
            random.choice(["Therefore,", "Hence,", "To conclude,"]),
            text[0].lower(),
            text[1:]
        )

    return text.lower().replace(".", "!").replace("?", "???")


fill_words = ("absolutely", "actual", "actually", "anyway", "apparently", "approximately", "basically", "certainly",
              "clearly", "completely", "definitely", "hopefully", "just", "largely", "particularly",  "really",
              "rather",  "totally", "very")


def remove_non_content_words_manipulate_punctuation(text: str) -> str:
    ret = text
    try:
        ret = " ".join(filter(lambda t: len(t) >= 2 and t not in fill_words,
                              word_tokenize(text=text, language="english", preserve_line=False)))
        logger.trace("Removed all the fill-words: \"{}\"->\"{}\"", text, ret)
    except LookupError:
        logger.opt(exception=False).warning("No fill-word-remove possible")

    return ret + random.choice(["", ".", "...", "!"])


def wordnet_changes_text(text: str, direction: Literal["more_concrete", "more_general", "similar"] = "similar",
                         change_threshold: float = .66, maximum_synsets_to_fix: int = 3) -> str:
    final_ret = ""
    try:
        for sent in pos_tag_sents([word_tokenize(text=s, language="english", preserve_line=False)
                                   for s in sent_tokenize(text=text, language="english")],
                                  tagset="universal"):
            logger.trace("Processing sentence: {}", " - ".join(map(lambda s: s[0], sent)))
            final_ret += " " if len(final_ret) >= 1 else ""
            for token, pos in sent:
                if pos in ["PRON", "ADP", "CONJ", "DET", "NUM", "PRT", "X"]:
                    logger.trace("There is nothing to change on \"{}\" - keep it!", token)
                    final_ret += " {}".format(token) if len(final_ret) >= 1 else token
                elif pos == ".":
                    final_ret += token
                else:
                    logger.trace("This word \"{}\" ({}) we may change...", token, pos)
                    if random.random() >= change_threshold:
                        logger.trace("Yes, the randomness says: change it!")
                        synsets = wordnet.synsets(
                            lemma=token,
                            pos=wordnet.NOUN if pos == "NOUN" else (wordnet.VERB if pos == "VERB" else
                                                                    (wordnet.ADJ if pos == "ADJ" else wordnet.ADV))
                        )

                        if len(synsets) == 0:
                            logger.trace("Word \"{}\" not found in WordNet (Version: {})", token, wordnet.get_version())
                            if pos == "ADJ":
                                final_ret += " {}{}".format("very "if direction != "more_general" else "", token)\
                                    if len(final_ret) >= 1 else token
                            elif pos == "VERB":
                                final_ret += " {}{}".format("{} ".format(random.choice(fill_words))
                                                            if direction != "more_general" else "", token) \
                                    if len(final_ret) >= 1 else token
                            else:
                                final_ret += " {}".format(token) if len(final_ret) >= 1 else token
                        elif 1 <= len(synsets) <= maximum_synsets_to_fix:
                            logger.trace("1 <= {} synsets <= {}", len(synsets), maximum_synsets_to_fix)
                            synset = random.choice([synsets[0]]*math.ceil(len(synsets)/2) + synsets)
                            logger.debug("We pick the synset \"{}\" - {}", synset.name(), synset.definition())
                            if direction == "more_general":
                                synset = random.choice(h) if len(h := synset.hypernyms()) >= 1 else synset
                                logger.debug("Successfully make a more general decision: \"{}\" -> {}",
                                             token, synset.name())
                            elif direction == "more_concrete":
                                synset = random.choice(h) if len(h := synset.hyponyms()) >= 1 else synset
                                logger.debug("Successfully make a more specific decision: \"{}\" -> {}",
                                             token, synset.name())

                            final_word = random.choice(synset.lemmas()).name().replace("_", " ")
                            logger.trace("Finally: \"{}\" --> \"{}\"", token, final_word)
                            final_ret += " {}".format(final_word) if len(final_ret) >= 1 else final_word
                        else:
                            logger.info("We have too much synsets (> {}): {}",
                                        maximum_synsets_to_fix, " + ".join(map(lambda s: s.name(), synsets)))
                            final_ret += " {}".format(token) if len(final_ret) >= 1 else token
                    else:
                        logger.trace("No change -- to high change_threshold of {}", change_threshold)
                        final_ret += " {}".format(token) if len(final_ret) >= 1 else token
    except LookupError:
        logger.opt(exception=True).warning("NLTK not complete - retry!")
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("universal_tagset")
        nltk.download("omw-1.4")
        return wordnet_changes_text(text=text, direction=direction, change_threshold=change_threshold,
                                    maximum_synsets_to_fix=maximum_synsets_to_fix)

    return final_ret if len(final_ret) >= 1 else text


def negate(text: str) -> str:
    global spacy_nlp

    logger.trace("This inverting-negation-code bases on "
                 "https://github.com/1itttlesheep/Factuality-evaluation-in-MT_Zuojun/"
                 "blob/3d24f4e1ed1419d57a26976a9d395ee74c8c92ae/checklist_generate/checklist/perturb.py")

    text = text.replace("e.g.", "for example")
    try:
        text_list = sent_tokenize(text)
    except LookupError:
        logger.opt(exception=False).warning("Some nltk-resources are missing to determine the amount of sentences")
        text_list = [text]

    if len(text_list) >= 2:
        logger.info("You have more than one sentence ({}). Negating several sentence may lead to incorrect results.",
                    len(text_list))

    if spacy_nlp is None:
        logger.warning("The spacy-pipeline isn't initialized yet - will change this now!")
        spacy_nlp = spacy.load("en_core_web_sm")

    def negate_sentence(text_sentence: str) -> str:
        def remove_negation(doc):
            """Removes negation from doc.
            This is experimental, may or may not work.
            Parameters
            ----------
            doc : spacy.token.Doc
                input
            Returns
            -------
            string
                With all negations removed
            """
            # This removes all negations in the doc. I should maybe add an option to remove just some.
            notzs = [i for i, z in enumerate(doc) if z.lemma_ == 'not' or z.dep_ == 'neg']
            new = []
            for notz in notzs:
                before = doc[notz - 1] if notz != 0 else None
                after = doc[notz + 1] if len(doc) > notz + 1 else None
                if (after and after.pos_ == 'PUNCT') or (before and before.text in ['or']):
                    continue
                new.append(notz)
            notzs = new
            if not notzs:
                return None
            f_ret = ''
            start = 0
            for i, notz in enumerate(notzs):
                id_start = notz
                to_add = ' '
                id_end = notz + 1
                before = doc[notz - 1] if notz != 0 else None
                after = doc[notz + 1] if len(doc) > notz + 1 else None
                if before and before.lemma_ in ['will', 'can', 'do']:
                    id_start = notz - 1
                    tenses = collections.Counter([x[0] for x in f_tenses(before.text)]).most_common(1)
                    tense = tenses[0][0] if len(tenses) else 'present'
                    p = tenses(before.text)
                    params = [tense, 3]
                    if p:
                        tmp = [x for x in p if x[0] == tense]
                        if tmp:
                            params = list(tmp[0])
                        else:
                            params = list(p[0])
                    to_add = ' ' + f_conjugate(before.lemma_, *params) + ' '
                if before and after and before.lemma_ == 'do' and after.pos_ == 'VERB':
                    id_start = notz - 1
                    tenses = collections.Counter([x[0] for x in f_tenses(before.text)]).most_common(1)
                    tense = tenses[0][0] if len(tenses) else 'present'
                    p = f_tenses(before.text)
                    params = [tense, 3]
                    if p:
                        tmp = [x for x in p if x[0] == tense]
                        if tmp:
                            params = list(tmp[0])
                        else:
                            params = list(p[0])
                    to_add = ' ' + f_conjugate(after.text, *params) + ' '
                    id_end = notz + 2
                f_ret += doc[start:id_start].text + to_add
                start = id_end
            f_ret += doc[id_end:].text
            return f_ret

        def add_negation(doc):
            """Adds negation to doc
            This is experimental, may or may not work. It also only works for specific parses.
            Parameters
            ----------
            doc : spacy.token.Doc
                input
            Returns
            -------
            string
                With negations added
            """
            for sentence in doc.sents:
                if len(sentence) < 3:
                    continue
                root_id = [x.i for x in sentence if x.dep_ == 'ROOT'][0]
                root = doc[root_id]
                if '?' in sentence.text and sentence[0].text.lower() == 'how':
                    continue
                if root.lemma_.lower() in ['thank', 'use']:
                    continue
                if root.pos_ not in ['VERB', 'AUX']:
                    continue
                neg = [True for x in sentence if x.dep_ == 'neg' and x.head.i == root_id]
                if neg:
                    continue
                if root.lemma_ == 'be':
                    if '?' in sentence.text:
                        continue
                    if root.text.lower() in ['is', 'was', 'were', 'am', 'are', '\'s', '\'re', '\'m']:
                        return doc[:root_id + 1].text + ' not ' + doc[root_id + 1:].text
                    else:
                        return doc[:root_id].text + ' not ' + doc[root_id:].text
                else:
                    aux = [x for x in sentence if x.dep_ in ['aux', 'auxpass'] and x.head.i == root_id]
                    if aux:
                        aux = aux[0]
                        if aux.lemma_.lower() in ['can', 'do', 'could', 'would', 'will', 'have', 'should']:
                            lemma = doc[aux.i].lemma_.lower()
                            if lemma == 'will':
                                fixed = 'won\'t'
                            elif lemma == 'have' and doc[aux.i].text in ['\'ve', '\'d']:
                                fixed = 'haven\'t' if doc[aux.i].text == '\'ve' else 'hadn\'t'
                            elif lemma == 'would' and doc[aux.i].text in ['\'d']:
                                fixed = 'wouldn\'t'
                            else:
                                fixed = doc[aux.i].text.rstrip('n') + 'n\'t' if lemma != 'will' else 'won\'t'
                            fixed = ' %s ' % fixed
                            return doc[:aux.i].text + fixed + doc[aux.i + 1:].text
                        return doc[:root_id].text + ' not ' + doc[root_id:].text
                    else:
                        # TODO: does, do, etc. Remover return None de cima
                        subj = [x for x in sentence if x.dep_ in ['csubj', 'nsubj']]
                        p = f_tenses(root.text)
                        tenses = collections.Counter([x[0] for x in f_tenses(root.text)]).most_common(1)
                        tense = tenses[0][0] if len(tenses) else 'present'
                        params = [tense, 3]
                        if p:
                            tmp = [x for x in p if x[0] == tense]
                            if tmp:
                                params = list(tmp[0])
                            else:
                                params = list(p[0])
                        if root.tag_ not in ['VBG']:
                            do = f_conjugate('do', *params) + 'n\'t'
                            new_root = f_conjugate(root.text, tense='infinitive')
                        else:
                            do = 'not'
                            new_root = root.text
                        return '%s %s %s %s' % (doc[:root_id].text, do, new_root, doc[root_id + 1:].text)

        s_nlp = spacy_nlp(text_sentence)
        logger.trace("Processed sentence \"{}\" ({})", text_sentence, s_nlp.vector)
        ret = remove_negation(s_nlp)
        if ret is None:
            logger.debug("Remove-negation failed in \"{}\" - no negation found, hence, add negation.", text_sentence)
            ret = add_negation(s_nlp)
            if ret is None:
                logger.warning("Failed to negate \"{}\" - jump to default case", text_sentence)
                return "It's not the case that {}{}".format(text_sentence[0].lower(), text_sentence[1:])

        return ret

    logger.trace("OK, let's start processing \"{}\"", text)

    return " ".join(map(lambda s: negate_sentence(s), text_list))
