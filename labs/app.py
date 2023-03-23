# Steps to write something:
# 0. Decide on a topic
# 1. Choose a genre
# 2. Choose a setting
# 3. Choose characters
# 4. Choose a title
# 5. Choose a plot
# 6. Write a paragraph
# 7. Write the rest of the paragraphs
# TODO: add motifs & themes
import random
import string
from time import sleep

import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI

INVENTIVE_LLM = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.9,
    presence_penalty=1.9,
    frequency_penalty=1.9,
)

UNINVENTIVE_LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

################################################################
# Write Something
################################################################

choose_genre_chain = LLMChain(
    llm=INVENTIVE_LLM,
    prompt=PromptTemplate.from_template(
        "You are a writer choosing a genre for your"
        " latest masterpiece."
        " You have decided on the topic of: {topic}."
        " Now, choose a genre:"
    ),
    output_key="genre",
)

choose_setting_chain = LLMChain(
    llm=INVENTIVE_LLM,
    prompt=PromptTemplate.from_template(
        "You are a writer creating a setting for your"
        " latest masterpiece."
        " You have decided on the topic of: {topic}."
        " You have decided on the genre of: {genre}."
        " Now, choose a 1 sentence setting:"
    ),
    output_key="setting",
)

choose_characters_chain = LLMChain(
    llm=INVENTIVE_LLM,
    prompt=PromptTemplate.from_template(
        "You are a writer creating characters for your"
        " latest masterpiece."
        " You have decided on the topic of: {topic}."
        " You have decided on the genre of: {genre}."
        " You have decided on the setting of: {setting}."
        " Now, choose 1-3 characters."
        " It should adhere to the following format:"
        " 1. character name : 1-sentence character description"
        " 2. character name : 1-sentence character description"
        " Begin!"
    ),
    output_key="characters",
)

choose_title_chain = LLMChain(
    llm=INVENTIVE_LLM,
    prompt=PromptTemplate.from_template(
        "You are a writer creating a title for your"
        " latest masterpiece."
        " You have decided on the topic of: {topic}."
        " You have decided on the genre of: {genre}."
        " You have decided on the setting of: {setting}."
        " You have decided on the characters of: {characters}."
        " Now, choose a title:"
    ),
    output_key="title",
)


choose_plot_chain = LLMChain(
    llm=INVENTIVE_LLM,
    prompt=PromptTemplate.from_template(
        "You are a writer creating a plot for your"
        " latest masterpiece."
        " You have decided on the topic of: {topic}."
        " You have decided on the genre of: {genre}."
        " You have decided on the setting of: {setting}."
        " You have decided on the characters of: {characters}."
        " You have decided on the title of: {title}."
        " Now, choose a 1 sentence plot:"
    ),
    output_key="plot",
)


choose_setup_from_topic_chain = SequentialChain(
    chains=[
        choose_genre_chain,
        choose_setting_chain,
        choose_characters_chain,
        choose_title_chain,
        choose_plot_chain,
    ],
    input_variables=["topic"],
    output_variables=["genre", "setting", "characters", "title", "plot"],
)

##########

write_first_paragraph_chain = LLMChain(
    llm=INVENTIVE_LLM,
    prompt=PromptTemplate.from_template(
        "You are a writer creating paragraphs for your"
        " latest masterpiece."
        "\nTopic: {topic}."
        "\nGenre: {genre}."
        "\nSetting: {setting}."
        "\nCharacters: {characters}."
        "\nTitle: {title}."
        "\nPlot: {plot}."
        " Now write the first paragraph of your masterpiece,"
        " and make it a strong, thought-provoking one."
    ),
    output_key="paragraph_0",
)


class WriteRemainingParagraphsChain(Chain):
    paragraph_writer_chain: LLMChain = LLMChain(
        llm=INVENTIVE_LLM,
        prompt=PromptTemplate.from_template(
            "You are a writer creating paragraphs for your"
            " latest masterpiece."
            "\nGenre: {genre}."
            "\nTitle: {title}."
            "\nCharacters: {characters}."
            " You're partially finished with your work."
            " Add the next paragraph to your masterpiece."
            " This will {not_insertion} be the last paragraph."
            " Make it a detailed one that progresses the piece."
            "\n\n{previous_paragraphs}"
        ),
    )

    @property
    def input_keys(self):
        return [
            "characters",
            "number_of_words",
            "topic",
            "genre",
            "title",
            "paragraph_0",
        ]

    @property
    def output_keys(self):
        return ["paragraphs"]

    @staticmethod
    def _not_insertion(paragraphs: str, number_of_words: int):
        """Insert a NOT if we don't think this will be the last paragraph."""
        if len(paragraphs.split()) > number_of_words - 150:
            return "NOT"
        return ""

    def _call(self, inputs: dict) -> dict:
        """Write paragraphs until the number of words is reached."""
        number_of_words = inputs["number_of_words"]
        new_paragraph = self.paragraph_writer_chain.run(
            {
                **inputs,
                "not_insertion": self._not_insertion(
                    inputs["paragraph_0"].strip(), inputs["number_of_words"]
                ),
                "previous_paragraphs": " . . . " + inputs["paragraph_0"].strip(),
            },
        )
        previous_paragraphs = (
            inputs["paragraph_0"].strip() + "\n\n" + new_paragraph.strip()
        )
        while len(previous_paragraphs.split()) < number_of_words:
            sleep(0.2)  # try to avoid rate limiting
            new_paragraph = self.paragraph_writer_chain.run(
                {
                    **inputs,
                    "not_insertion": self._not_insertion(
                        previous_paragraphs.strip(), inputs["number_of_words"]
                    ),
                    "previous_paragraphs": " . . . "
                    + previous_paragraphs[-10_000:],  # don't hit token limits
                }
            )
            previous_paragraphs += "\n\n" + new_paragraph.strip()

        return {
            "paragraphs": previous_paragraphs,
        }


write_paragraphs_chain = SequentialChain(
    chains=[
        write_first_paragraph_chain,
        WriteRemainingParagraphsChain(),
    ],
    input_variables=[
        "topic",
        "genre",
        "setting",
        "characters",
        "title",
        "plot",
        "number_of_words",
    ],
)


def write_something():
    """Write something."""
    st.title("Write Something")
    st.write("This app will help you write something.")

    form = st.form(key="write_something_form")
    topic = form.text_input("Give me something to write about.", max_chars=100)
    number_of_words = form.slider(
        "How many words would you like to write?",
        min_value=0,
        max_value=1000,
        value=250,
        step=50,
    )
    submitted = form.form_submit_button("Write Something")

    if not topic:
        st.session_state = {}
        st.stop()
    if not submitted:
        st.session_state = {}
        st.stop()

    with st.spinner("Writing the setup..."):
        # Get the setup from the session state or generate it.
        setup = st.session_state.get("setup") or choose_setup_from_topic_chain(topic)
        st.session_state["setup"] = setup

    for key, value in setup.items():
        st.markdown("#### " + key.replace("_", " ").title())
        st.write(value)

    with st.spinner("Writing the actual 'something'..."):
        paragraphs = (
            st.session_state.get("paragraphs")
            or write_paragraphs_chain(
                {**setup, "number_of_words": number_of_words},
            )["paragraphs"]
        )
        st.session_state["paragraphs"] = paragraphs

    st.subheader("Here is your masterpiece:")
    st.write(paragraphs)

    st.download_button(
        label="Download Masterpiece",
        data="\n".join([setup["title"], paragraphs]),
        file_name=f"{' '.join(setup['title'].strip().split()).strip().replace(' ', '_').lower()}.txt",
    )


################################################################
# Human or AI?
################################################################

_generate_paired_example_chain = LLMChain(
    llm=INVENTIVE_LLM,
    prompt=PromptTemplate.from_template(
        "You are an expert impersonator of a human."
        " You are trying to fool a human into thinking you are a human."
        " I will give you some text that was written by a human."
        " Using that text, write your own counter-example that mimics"
        "the human's in length, style, and content."
        " Here is the text:\n{text_in_question}\n"
        " Begin!"
    ),
    output_key="generated_text",
)


def _shuffle_examples(inputs: dict) -> dict:
    """Shuffle the examples so that the AI is not always first."""
    if random.random() > 0.5:
        return {
            "example_1": inputs["text_in_question"],
            "example_2": inputs["generated_text"]
            .strip()
            .strip(string.punctuation)
            .strip(),
            "input_text_loc": "1",
            "generated_text_": inputs["generated_text"]
            .strip()
            .strip(string.punctuation)
            .strip(),
            "text_in_question_": inputs["text_in_question"],
            # **inputs,
        }
    return {
        "example_1": inputs["generated_text"].strip().strip(string.punctuation).strip(),
        "example_2": inputs["text_in_question"],
        "input_text_loc": "2",
        "generated_text_": inputs["generated_text"]
        .strip()
        .strip(string.punctuation)
        .strip(),
        "text_in_question_": inputs["text_in_question"],
        # **inputs,
    }


_shuffle_examples_chain = TransformChain(
    input_variables=["text_in_question", "generated_text"],
    output_variables=[
        "example_1",
        "input_text_loc",
        "example_2",
        "generated_text_",
        "text_in_question_",
    ],
    transform=_shuffle_examples,
)


_decide_which_is_ai_chain = LLMChain(
    llm=UNINVENTIVE_LLM,
    prompt=PromptTemplate.from_template(
        "You are given 2 examples of text."
        " One was written by a human and the other was written by an AI."
        " You must decide which is which. Here are the examples:"
        "\n[1] {example_1}\n[2] {example_2}\n"
        " Now, which is the AI-written example? [1] or [2]?"
        " If it is difficult to decide, just say [0]."
    ),
    output_key="prediction",
)

single_example_chain = SequentialChain(
    chains=[
        _generate_paired_example_chain,
        _shuffle_examples_chain,
        _decide_which_is_ai_chain,
    ],
    input_variables=["text_in_question"],
    output_variables=[
        "prediction",
        "generated_text_",
        "text_in_question_",
        "input_text_loc",
    ],
)


def _was_input_text_ai(outputs: dict) -> bool:
    """Did the model predict the input-text as ai?"""
    input_text_loc = "[" + outputs["input_text_loc"].strip() + "]"
    try:
        input_text_loc_index = outputs["prediction"].index(input_text_loc)
    except ValueError:
        input_text_loc_index = 100_000

    other_text_loc = "[1]" if outputs["input_text_loc"] == "2" else "[2]"
    try:
        other_text_loc_index = outputs["prediction"].index(other_text_loc)
    except ValueError:
        other_text_loc_index = 100_000

    zero_in = "[0]" in outputs["prediction"]

    if zero_in and input_text_loc_index == other_text_loc_index:
        return None
    elif input_text_loc_index > other_text_loc_index:
        return True
    else:
        return False


def human_or_ai():
    st.title("Human or AI?")

    text_in_question = st.text_area("Please input some text here.", max_chars=2_500)
    run = st.button("Run")

    if not text_in_question:
        st.stop()
    elif run and not text_in_question:
        st.warning("Please input some text.")
        st.stop()
    elif text_in_question != st.session_state.get("text_in_question"):
        st.session_state = {"text_in_question": text_in_question}

    counts_for_ai = 0
    total_runs = 3
    with st.spinner("Generating predictions . . ."):
        for i in range(total_runs):
            with st.spinner(f"Generating prediction {i+1} . . ."):
                outputs = st.session_state.get(f"outputs_{i}") or single_example_chain(
                    {"text_in_question": text_in_question}
                )
                st.session_state[f"outputs_{i}"] = outputs
                # print(outputs)

            st.markdown(f"#### Generated Impersonation {i+1}:")
            st.markdown(":blue[" + outputs["generated_text_"].strip() + "]")
            input_text_was_ai = st.session_state.get(
                f"input_text_{i}_was_ai"
            ) or _was_input_text_ai(outputs)
            st.session_state[f"input_text_{i}_was_ai"] = input_text_was_ai
            if input_text_was_ai is None:
                st.write("The AI could not decide which was AI-written.")
                total_runs -= 1
            elif input_text_was_ai:
                st.write("The AI predicted the input was AI-written.")
                counts_for_ai += 1
            else:
                st.write("The AI predicted the input was Human-written.")
    result = counts_for_ai / total_runs
    st.markdown(
        f":red[The AI predicted the input was AI-written {result:.0%} of the time.]"
    )
    st.markdown(
        "### Your Text Was Written By An AI!"
        if result > 0.5
        else "### Your Text Was Written By A Human!"
    )


if __name__ == "__main__":
    with st.sidebar:
        choices = {
            "Write Something": write_something,
            "Human or AI?": human_or_ai,
        }
        app_choice = st.radio(
            "**Choose an app:**",
            options=list(choices.keys()),
        )
    if app_choice:
        if "app_choice" not in st.session_state:
            st.session_state["app_choice"] = app_choice
        elif st.session_state["app_choice"] != app_choice:
            st.session_state = {"app_choice": app_choice}

        choices[app_choice]()
