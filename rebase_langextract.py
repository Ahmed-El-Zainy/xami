import langextract as lx
import textwrap
import os
from getpass import getpass
from IPython.display import display, HTML


# Set up your Gemini API key
# Get your key from: https://aistudio.google.com/app/apikey
if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = getpass('Enter your Gemini API key: ')

# Define the extraction task with an improved prompt for Q&A mapping
prompt = textwrap.dedent("""\
    Extract related sentecnce, questions, answer choices, and correct answers from Arabic educational text.
    Also identify and explicitly state relationships that connect questions to their choices and correct answers.

    Rules:
    1. Extract exact text for questions, choices, and answers. Do not paraphrase.
    2. Identify and extract the question number/marker (like [١], ١], etc.).
    3. Extract all available multiple choice options (أ-, ب-, ج-, د-, etc.).
    4. Find and extract the correct answer, including its number/marker and the correct choice letter.
    5. Create clear relationships between each question, its choices, and its correct answer.

    Focus on preserving Arabic text formatting and numbering systems.

    Example:
    Input Text:
    --- Page 1 ---
    
    ## ١] نفهم من الفقرة الأولى أن السبب المباشر لحدوث الزلازل هو:

    أ- التشققات الأرضية الداخلية.
    ب- التقلصات والضغوط الكبيرة في باطن الأرض.
    ج- الإزاحة العمودية أو الأفقيه بين الصخور.
    د- الكميات الهائلة من الطاقة المتولدة من باطن الأرض.

    --- Page 10 ---
    الإجابة
    [١] د- الكميات الهائلة من الطاقة المتولدة من باطن الأرض.

    Expected Extractions:
    [
      {
        "extraction_class": "question",
        "extraction_text": "١] نفهم من الفقرة الأولى أن السبب المباشر لحدوث الزلازل هو:",
        "attributes": {
          "question_number": "١"
        }
      },
      {
        "extraction_class": "choices",
        "extraction_text": "أ- التشققات الأرضية الداخلية.\nب- التقلصات والضغوط الكبيرة في باطن الأرض.\nج- الإزاحة العمودية أو الأفقيه بين الصخور.\nد- الكميات الهائلة من الطاقة المتولدة من باطن الأرض.",
        "attributes": {
          "question_number": "١",
          "choice_labels": ["أ", "ب", "ج", "د"]
        }
      },
      {
        "extraction_class": "answer",
        "extraction_text": "[١] د- الكميات الهائلة من الطاقة المتولدة من باطن الأرض.",
        "attributes": {
          "question_number": "١",
          "correct_choice": "د"
        }
      },
      {
        "extraction_class": "relationship",
        "extraction_text": "Question ١ is mapped to choices أ-د and answer [١] د",
        "attributes": {
          "type": "question_answer_mapping",
          "question_number": "١",
          "correct_answer_choice": "د"
        }
      }
    ]
    """)



# Provide comprehensive examples (using the improved structure)
examples = [
    lx.data.ExampleData(
        text="""

--- Page 1 ---
اقرأ ثم أجب:

١. تعرف الهزات على أنها ظاهرة كونية فيزيائية بالغة التعقيد، تظهر كحركات عشوائية للقشرة الأرضية على شكل ارتعاش وتحرك وتموج عنيف، وذلك نتيجة إطلاق كميات هائلة من الطاقة من باطن الأرض، وهذه الطاقة تتولد نتيجة لإزاحة عمودية أو أفقية بين صخور الأرض عبر الصدوع التي تحدث لتعرضها المستمر للتقلصات والضغوط الكبيرة.

٢. ويعاظم تأثير الهزات في الأراضي الضاغيفة خصوصا في الرواسب الرملية والطينية حديثة التكوين، ويعلل ذلك بأن هذه الرواسب تهتز بعنف بسبب انخفاض معامل مرونتها وصلابتها وعدم مقدرتها على تخفيض التأثير التسارعي الذي تتعرض له الحبيبات بفعل الزلازل.

٣. وفي حالة الزلازل التي تقع مراكزها السطحية في قاع البحار أو المحيطات فقد تؤدي إلى حدوث أمواج مائية ضخمة جدا تسمى التسونامي وهي كلمة يابانية معناها أمواج الموانئ أو الخلجان، إذ تؤدي الاهتزازات المصاحبة لحدوث الزلازل إلى تكون هذه الأمواج، وقد تصل سرعتها إلى ٨٠٠ كم/ساعة، وذلك نتيجة لانزلاق صفائح القشرة الأرضية عموديا بعضها على بعض، وما يجدر ذكره هنا أن الزلازل التي تنشأ عن انزلاقا أفقية في الصفائح لا تؤدي إلى تكون أمواج التسونامي.

أ. وعموما تنشأ الزلازل التكتونية نتيجة للحركة النسبية للصفائح - القطع التي تتشكل منها القشرة الأرضية. إذ تتحرك القارات مبتعدة أو مقتربة بعضها من بعض مشكلة إجهادات ضغط وشد بعضها على بعض ، ويبدأ تراكم الإجهادات الداخلية في طبقات الصخور الواقعة على حدود الصفائح المتحركة، وعندما تصبح قيم الإجهادات المتراكمة أكبر من قيمة الإجهادات القصوى التي يمكن أن تتحملها الصخور تحصل كسور وتحركات فجائية لطبقات الصخور، ما يؤدي إلى إطلاق كمية هائلة من الطاقة المتراكمة، حيث تنتقل هذه الطاقة على شكل موجات زلزالية في جميع الاتجاهات.

[١] نفهم من الفقرة الأولى أن السبب المباشر لحدوث الزلازل هو:

أ- التشققات الأرضية الداخلية.

ب- التقلصات والضغوط الكبيرة في باطن الأرض.

ج- الإزاحة العمودية أو الأفقية بين الصخور.

د- الكميات الهائلة من الطاقة المتولدة من باطن الأرض.

01224061317

--- Page 2 ---
[١] نستنتج من الفقرة الثانية أن تأثير الزلازل على الأرض الصلبة:

أ- أقوى بسبب انخفاض معامل مروناتها وصلابتها.

ب- أقل بسبب انخفاض معامل مروناتها وصلابتها.

ج- أقوى بسبب ارتفاع معامل مروناتها وصلابتها.

د- أقل بسبب ارتفاع معامل مروناتها وصلابتها.

[٢] يفهم من الفقرة الثالثة أن التسونامي تنشأ بسبب:

أ- الانزلاقات الأفقية في صفائح القشرة الأرضية.

ب- أمواج الموانئ والخلجان المضطربة والشديدة.

ج- انزلاق صفائح القشرة الأرضية عموديًا بعضها على بعض.

د- سرعة الرياح التي تصل إلى ٨٠٠ كم/ساعة.

--- Page 10 ---
الإجابة

[١] د- الكميات الهائلة من الطاقة المتولدة من باطن الأرض.

[٢] د- أقل بسبب ارتفاع معامل مرونتها وصلابتها.

        """,
        ## Q1
        extractions=[
            lx.data.Extractions(
                extraction_class="sentence",
                extractions_text="""
١. تعرف الهزات على أنها ظاهرة كونية فيزيائية بالغة التعقيد، تظهر كحركات عشوائية للقشرة الأرضية على شكل ارتعاش وتحرك وتموج عنيف، وذلك نتيجة إطلاق كميات هائلة من الطاقة من باطن الأرض، وهذه الطاقة تتولد نتيجة لإزاحة عمودية أو أفقية بين صخور الأرض عبر الصدوع التي تحدث لتعرضها المستمر للتقلصات والضغوط الكبيرة.

٢. ويعاظم تأثير الهزات في الأراضي الضاغيفة خصوصا في الرواسب الرملية والطينية حديثة التكوين، ويعلل ذلك بأن هذه الرواسب تهتز بعنف بسبب انخفاض معامل مرونتها وصلابتها وعدم مقدرتها على تخفيض التأثير التسارعي الذي تتعرض له الحبيبات بفعل الزلازل.

٣. وفي حالة الزلازل التي تقع مراكزها السطحية في قاع البحار أو المحيطات فقد تؤدي إلى حدوث أمواج مائية ضخمة جدا تسمى التسونامي وهي كلمة يابانية معناها أمواج الموانئ أو الخلجان، إذ تؤدي الاهتزازات المصاحبة لحدوث الزلازل إلى تكون هذه الأمواج، وقد تصل سرعتها إلى ٨٠٠ كم/ساعة، وذلك نتيجة لانزلاق صفائح القشرة الأرضية عموديا بعضها على بعض، وما يجدر ذكره هنا أن الزلازل التي تنشأ عن انزلاقا أفقية في الصفائح لا تؤدي إلى تكون أمواج التسونامي.

أ. وعموما تنشأ الزلازل التكتونية نتيجة للحركة النسبية للصفائح - القطع التي تتشكل منها القشرة الأرضية. إذ تتحرك القارات مبتعدة أو مقتربة بعضها من بعض مشكلة إجهادات ضغط وشد بعضها على بعض ، ويبدأ تراكم الإجهادات الداخلية في طبقات الصخور الواقعة على حدود الصفائح المتحركة، وعندما تصبح قيم الإجهادات المتراكمة أكبر من قيمة الإجهادات القصوى التي يمكن أن تتحملها الصخور تحصل كسور وتحركات فجائية لطبقات الصخور، ما يؤدي إلى إطلاق كمية هائلة من الطاقة المتراكمة، حيث تنتقل هذه الطاقة على شكل موجات زلزالية في جميع الاتجاهات.
""" ,
                attributes={"type": "sentence_text",}
            ),
            lx.data.Extraction(
                extraction_class="question",
                extraction_text="١] نفهم من الفقرة الأولى أن السبب المباشر لحدوث الزلازل هو:",
                attributes={"question_number": "١"}
            ),
            lx.data.Extraction(
                extraction_class="choices",
                extraction_text="أ- التشققات الأرضية الداخلية.\nب- التقلصات والضغوط الكبيرة في باطن الأرض.\nج- الإزاحة العمودية أو الأفقيه بين الصخور.\nد- الكميات الهائلة من الطاقة المتولدة من باطن الأرض.",
                attributes={"question_number": "١", "choice_labels": ["أ", "ب", "ج", "د"]}
            ),
            lx.data.Extraction(
                extraction_class="answer",
                extraction_text="[١] د- الكميات الهائلة من الطاقة المتولدة من باطن الأرض.",
                attributes={"question_number": "١", "correct_choice": "د"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Question 1 is mapped to choices أ-د and answer [1] د",
                attributes={"type": "question_answer_mapping", "question_number": "1", "correct_answer_choice": "د"}
            ),

            #### Q2
            lx.data.Extraction(
                extraction_class="question",
                extraction_text="[١] نستنتج من الفقرة الثانية أن تأثير الزلازل على الأرض الصلبة:",
                attributes ={"question_number": "٢"}

            ),
            lx.data.Extraction(
                extraction_class="choices",
                extraction_text="أ- الانزلاقات الأفقية في صفائح القشرة الأرضية.\nب- أمواج الموانئ والخلجان المضطربة والشديدة.\n.ج- انزلاق صفائح القشرة الأرضية عموديًا بعضها على بعض.\nد- سرعة الرياح التي تصل إلى ٨٠٠ كم/ساعة.",
                attributes={"question_number": "١", "choice_labels": ["أ", "ب", "ج", "د"]}
            ),
            lx.data.Extraction(
                extraction_class="answer",
                extraction_text="[٢] د- أقل بسبب ارتفاع معامل مرونتها وصلابتها.",
                attributes={"question_number": "٢", "correct_choice": "د"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Question 2 is mapped to choices أ-د and answer [2] د",
                attributes={"type": "question_answer_mapping", "question_number": "2", "correct_answer_choice": "د"}
            ),
        ]




    )
]


# Function to process your text file
def extract_qa_from_file(file_path, api_key=None):
    """
    Extract questions, choices, and answers from an Arabic text file
    """

    # Read the file contents first
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()

        if not text_content.strip():
            raise ValueError("File is empty or contains no readable content")

        print(f"📖 Successfully read file: {len(text_content)} characters")
        print(f"📝 First 200 characters: {text_content[:200]}...")

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except UnicodeDecodeError:
        # Try different encodings
        for encoding in ['utf-8-sig', 'cp1256', 'iso-8859-6']:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text_content = file.read()
                print(f"📖 Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError("Could not decode file with common Arabic encodings")

    # Extract from the actual text content, not the file path
    result = lx.extract(
        text_or_documents=text_content,  # Pass the actual content, not file path
        prompt_description=prompt,
        examples=examples,
        model_id="gemini-2.0-flash-exp",  # Using the latest model
        extraction_passes=3,  # Multiple passes for better recall
        max_workers=10,  # Parallel processing
        max_char_buffer=4000,  # Increased buffer for larger documents
        api_key=api_key
    )

    return result

# Function to analyze and display results
def analyze_results(result):
    """
    Analyze and display extracted Q&A data
    """
    questions = []
    choices = []
    answers = []
    relationships = []

    if result and result.extractions:
        for extraction in result.extractions:
            if extraction: # Check if extraction is not None
                if extraction.extraction_class == "question":
                    questions.append(extraction)
                elif extraction.extraction_class == "choices":
                    choices.append(extraction)
                elif extraction.extraction_class == "answer":
                    answers.append(extraction)
                elif extraction.extraction_class == "relationship":
                    relationships.append(extraction)

    print(f"📊 Extraction Summary:")
    print(f"   Questions: {len(questions)}")
    print(f"   Choice sets: {len(choices)}")
    print(f"   Answers: {len(answers)}")
    print(f"   Relationships: {len(relationships)}")
    print("=" * 50)

    # Display questions with their choices and answers
    for i, question in enumerate(questions, 1):
        q_num = question.attributes.get("question_number", str(i))
        print(f"\n🔸 Question {q_num}:")
        print(f"   {question.extraction_text.strip()}")

        # Find corresponding choices
        q_choices = [c for c in choices if c.attributes and c.attributes.get("question_number") == q_num]
        if q_choices:
            print(f"\n   📝 Choices:")
            # Split choices by lines and print
            for choice_line in q_choices[0].extraction_text.split('\n'):
                if choice_line.strip():
                    print(f"      {choice_line.strip()}")

        # Find corresponding answer
        q_answers = [a for a in answers if a.attributes and a.attributes.get("question_number") == q_num]
        if q_answers:
            correct_choice = q_answers[0].attributes.get("correct_choice", "")
            print(f"\n   ✅ Correct Answer: {correct_choice}")
            print(f"      {q_answers[0].extraction_text.strip()}")

        print("-" * 30)

# Main execution function
def main(file_path, api_key=None):
    """
    Main function to process Arabic Q&A extraction
    """
    try:
        # Extract Q&A data
        print("🚀 Starting extraction...")
        result = extract_qa_from_file(file_path, api_key)

        # Analyze results
        analyze_results(result)

        # Save results
        output_filename = f"{os.path.basename(file_path).split('.')[0]}_qa_extracted.jsonl"
        lx.io.save_annotated_documents([result], output_name=output_filename, output_dir=".")
        print(f"\n💾 Results saved to: {output_filename}")

        # Generate visualization
        html_content = lx.visualize(output_filename)
        html_filename = f"{os.path.basename(file_path).split('.')[0]}_qa_extracted_visualization.html"

        with open(html_filename, "w", encoding='utf-8') as f:
            if hasattr(html_content, 'data'):
                f.write(html_content.data)
            else:
                f.write(str(html_content))

        print(f"📊 Visualization saved to: {html_filename}")

        # Optional: Display interactive visualization in notebook
        print("\n📈 Interactive visualization:")
        display(html_content)


        return result

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please check if the file path is correct.")
        return None
    except ValueError as e:
         print(f"❌ Error: {e}")
         print("The input file seems to be empty or contains no readable content.")
         return None
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")
        return None

# Usage example
if __name__=="__main__":
    file_path = "/content/output/test_1_extracted_text.txt"
    api_key = os.environ.get("GEMINI_API_KEY")
    # Run extraction
    main(file_path, api_key)