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

from time import sleep

import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.base import Chain
from langchain.llms import OpenAI

##########

LLM = OpenAI(temperature=0.4, presence_penalty=1.5, frequency_penalty=1.5)

# describe_topic_chain = LLMChain(
#     llm=LLM,
#     prompt=PromptTemplate.from_template(
#         "You are a writer describing in detail the topic"
#         " for your latest masterpiece."
#         " Describe the topic: {topic}"
#     ),
# )

choose_genre_chain = LLMChain(
    llm=LLM,
    prompt=PromptTemplate.from_template(
        "You are a writer choosing a genre for your"
        " latest masterpiece."
        " You have decided on the topic of: {topic}."
        " Now, choose a genre:"
    ),
    output_key="genre",
)

choose_setting_chain = LLMChain(
    llm=LLM,
    prompt=PromptTemplate.from_template(
        "You are a writer creating a setting for your"
        " latest masterpiece."
        " You have decided on the topic of: {topic}."
        " You have decided on the genre of: {genre}."
        " Now, choose a 2 sentence setting:"
    ),
    output_key="setting",
)

choose_characters_chain = LLMChain(
    llm=LLM,
    prompt=PromptTemplate.from_template(
        "You are a writer creating characters for your"
        " latest masterpiece."
        " You have decided on the topic of: {topic}."
        " You have decided on the genre of: {genre}."
        " You have decided on the setting of: {setting}."
        " Now, choose 2-7 characters."
        " It should adhere to the following format:"
        " 1. character name : 1-sentence character description"
        " 2. character name : 1-sentence character description"
        " Begin!"
    ),
    output_key="characters",
)

choose_title_chain = LLMChain(
    llm=LLM,
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
    llm=LLM,
    prompt=PromptTemplate.from_template(
        "You are a writer creating a plot for your"
        " latest masterpiece."
        " You have decided on the topic of: {topic}."
        " You have decided on the genre of: {genre}."
        " You have decided on the setting of: {setting}."
        " You have decided on the characters of: {characters}."
        " You have decided on the title of: {title}."
        " Now, choose a 2 sentence plot:"
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
    llm=LLM,
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
        llm=LLM,
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


def main():
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


if __name__ == "__main__":
    main()
