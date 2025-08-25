### Prompts

---
- اختيارات 
---
```json

أنت خبير تقويم تربوي لمادة الفيزياء للمرحلة الثانوية.  \\
مهمتك: إنشاء جميع أنواع الأسئلة (اختيار من متعدد، صح/خطأ، إجابة قصيرة، إلخ) من النص المرفق فقط، مع **دعم كامل لعرض المعادلات والدوال الفيزيائية في HTML**.

\[متطلبات عامة]


{
  "num_questions": 10,
  "topics": [
    {"topic_name": "الفصل الأول", "topic_id": 2},
    {"topic_name": "الفصل الثاني", "topic_id": 3},
    {"topic_name": "الفصل الثالث", "topic_id": 4}
  ],
  "topic_details": "تفاصيل عن كل فصل",
  "subject_id": 2,
  "choices_index": [0, 1, 2, 3, 4]
}
```

\[مخطط الإخراج النهائي]

```
[
{
  "id": "<معرّف السؤال>",
  "question": "<span>نص السؤال (استخدم وسوم HTML لعرض الدوال والمعادلات الفيزيائية، مثل <math><mi>F</mi><mo>=</mo><mi>m</mi><mo>×</mo><mi>a</mi></math>)</span>",
  "choice1": "<span class=\"col-xs-11 ans\">الاختيار الأول</span>",
  "choice2": "<span class=\"col-xs-11 ans\">الاختيار الثاني</span>",
  "choice3": "<span class=\"col-xs-11 ans\">الاختيار الثالث</span>",
  "choice4": "<span class=\"col-xs-11 ans\">الاختيار الرابع</span>",
  "choice5": null,
  "choice6": null,
  "choice7": null,
  "choice8": null,
  "choice9": null,
  "choice10": null,
  "choice11": null,
  "choice12": null,
  "correct_answer": "<index الصحيح>",
  "level": "0",
  "subject_id": "0",
  "topic_id": "<رقم الموضوع>",
  "image": "",
  "difficulty": "سهل",
  "topic_dbr": "<سلسلة الترميز>"
},
{...},
{...}
]
```

\[إرشادات تصميم المعادلات والدوال الفيزيائية]

1. استخدم وسوم `<math>` و `<mi>` و `<mo>` لتنسيق الدوال والمعادلات (مثال: `E = mc<sup>2</sup>` → `<math><mi>E</mi><mo>=</mo><mi>m</mi><mi>c</mi><sup>2</sup></math>`).
2. استخدم `<sub>` و `<sup>` للرموز السفلية والعلوية.
3. استخدم `<var>` لتحديد الرموز المتغيرة.
4. جميع الأسئلة متوافقة مع HTML للعرض المباشر.
5. جميع القوالب (MCQ، صح/خطأ، مقالي) تتبع نفس هيكل JSON.
6. صياغة الأسئلة بالعربية الفصحى، واضحة ومباشرة.

\[إرشادات توليد الأسئلة]
7\. كل سؤال له 4 بدائل، بديل واحد صحيح.
8\. غلّف السؤال بـ `<span>` والاختيارات بـ `<span class=\"col-xs-11 ans\">…</span>`.
9\. استخدم `<math>` أو `<sup>/<sub>` للمعادلات.
10\. `correct_answer` هو رقم البديل الصحيح كنص.
11\. الأرقام: عربية (0-9).
12\. لا تستخدم "كل ما سبق" إلا لو ورد بالنص.
13\. وزّع المستويات (سهل/متوسط/متقدم).
14\. لا تضف أي معلومة خارج النص.
15\. لا تكرر أي إجابة صحيحة أو تعطي أكثر من إجابة صحيحة.
16\. تأكد أن جميع المعادلات قابلة للقراءة.
17\. لا تضع أي توضيح للإجابات في المخرجات.

---
---

### Examples Multiple Choice
---
أهلاً بك! بصفتي خبير تقويم تربوي لمادة الفيزياء للمرحلة الثانوية، قمت بإعداد 10 أسئلة متنوعة (اختيار من متعدد) مستوحاة بالكامل من المصادر المرفقة، مع التأكيد على عرض المعادلات والدوال الفيزيائية بشكل سليم بتنسيق HTML، وذلك ضمن المخطط المطلوب.

إليك الأسئلة:

```json
[
  {
    "id": "PHY-U1-CH1-Q1",
    "question": "<span>وفقًا لقانون أوم، تُعطى مقاومة الموصل (<math><mi>R</mi></math>) بدلالة فرق الجهد (<math><mi>V</mi></math>) والتيار (<math><mi>I</mi></math>) بالعلاقة:</span>",
    "choice1": "<span class=\"col-xs-11 ans\"><math><mi>R</mi><mo>=</mo><mi>V</mi><mo>×</mo><mi>I</mi></math></span>",
    "choice2": "<span class=\"col-xs-11 ans\"><math><mi>R</mi><mo>=</mo><mfrac><mi>I</mi><mi>V</mi></mfrac></math></span>",
    "choice3": "<span class=\"col-xs-11 ans\"><math><mi>R</mi><mo>=</mo><mfrac><mi>V</mi><mi>I</mi></mfrac></math></span>",
    "choice4": "<span class=\"col-xs-11 ans\"><math><mi>R</mi><mo>=</mo><msup><mi>V</mi><mn>2</mn></msup><mo>×</mo><mi>I</mi></math></span>",
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "3",
    "level": "0",
    "subject_id": "0",
    "topic_id": "101",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "قانون أوم"
  },
  {
    "id": "PHY-U1-CH1-Q2",
    "question": "<span>عند توصيل <var>N</var> من المقاومات المتماثلة قيمة الواحدة منها <math><mi>R</mi></math> على التوالي، فإن المقاومة المكافئة (<math><msup><mi>R</mi><mo>′</mo></msup></math>) للمجموعة تكون:</span>",
    "choice1": "<span class=\"col-xs-11 ans\"><math><msup><mi>R</mi><mo>′</mo></msup><mo>=</mo><mfrac><mi>R</mi><mi>N</mi></mfrac></math></span>",
    "choice2": "<span class=\"col-xs-11 ans\"><math><msup><mi>R</mi><mo>′</mo></msup><mo>=</mo><mi>R</mi><mo>+</mo><mi>N</mi></math></span>",
    "choice3": "<span class=\"col-xs-11 ans\"><math><msup><mi>R</mi><mo>′</mo></msup><mo>=</mo><mi>N</mi><mi>R</mi></math></span>",
    "choice4": "<span class=\"col-xs-11 ans\"><math><msup><mi>R</mi><mo>′</mo></msup><mo>=</mo><mi>R</mi><mo>×</mo><msup><mi>N</mi><mn>2</mn></msup></math></span>",
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "3",
    "level": "0",
    "subject_id": "0",
    "topic_id": "102",
    "image": "",
    "difficulty": "متوسط",
    "topic_dbr": "توصيل المقاومات على التوالي"
  },
  {
    "id": "PHY-U1-CH2-Q3",
    "question": "<span>أي من العوامل التالية يؤدي إلى زيادة كثافة الفيض المغناطيسي (<math><mi>B</mi></math>) الناشئ عن تيار كهربي يمر في سلك مستقيم؟</span>",
    "choice1": "<span class=\"col-xs-11 ans\">الابتعاد عن السلك</span>",
    "choice2": "<span class=\"col-xs-11 ans\">نقص شدة التيار الكهربي</span>",
    "choice3": "<span class=\"col-xs-11 ans\">زيادة شدة التيار الكهربي</span>",
    "choice4": "<span class=\"col-xs-11 ans\">تقليل النفاذية المغناطيسية للوسط</span>",
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "3",
    "level": "0",
    "subject_id": "0",
    "topic_id": "103",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "كثافة الفيض المغناطيسي لسلك مستقيم"
  },
  {
    "id": "PHY-U1-CH2-Q4",
    "question": "<span>تُستخدم قاعدة اليد اليسرى لفلمنج لتحديد:</span>",
    "choice1": "<span class=\"col-xs-11 ans\">اتجاه التيار الكهربي في الدائرة</span>",
    "choice2": "<span class=\"col-xs-11 ans\">اتجاه المجال المغناطيسي</span>",
    "choice3": "<span class=\"col-xs-11 ans\">اتجاه القوة المؤثرة على سلك يمر به تيار في مجال مغناطيسي</span>",
    "choice4": "<span class=\"col-xs-11 ans\">قيمة المقاومة الكهربية</span>",
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "3",
    "level": "0",
    "subject_id": "0",
    "topic_id": "104",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "قاعدة اليد اليسرى لفلمنج"
  },
  {
    "id": "PHY-U1-CH3-Q5",
    "question": "<span>وفقًا لقانون فاراداي في الحث الكهرومغناطيسي، تُعطى القوة الدافعة الكهربية المستحثة (<math><mi>e</mi><mi>m</mi><mi>f</mi></math>) في ملف ذي عدد لفات (<math><mi>N</mi></math>) نتيجة لتغير الفيض المغناطيسي (<math><mi>Δ</mi><mi>Φ</mi></math>) خلال فترة زمنية (<math><mi>Δ</mi><mi>t</mi></math>) بالعلاقة:</span>",
    "choice1": "<span class=\"col-xs-11 ans\"><math><mi>e</mi><mi>m</mi><mi>f</mi><mo>=</mo><mi>N</mi><mfrac><mrow><mi>Δ</mi><mi>Φ</mi></mrow><mrow><mi>Δ</mi><mi>t</mi></mrow></mfrac></math></span>",
    "choice2": "<span class=\"col-xs-11 ans\"><math><mi>e</mi><mi>m</mi><mi>f</mi><mo>=</mo><mo>−</mo><mi>N</mi><mfrac><mrow><mi>Δ</mi><mi>Φ</mi></mrow><mrow><mi>Δ</mi><mi>t</mi></mrow></mfrac></math></span>",
    "choice3": "<span class=\"col-xs-11 ans\"><math><mi>e</mi><mi>m</mi><mi>f</mi><mo>=</mo><mfrac><mn>1</mn><mi>N</mi></mfrac><mfrac><mrow><mi>Δ</mi><mi>Φ</mi></mrow><mrow><mi>Δ</mi><mi>t</mi></mrow></mfrac></math></span>",
    "choice4": "<span class=\"col-xs-11 ans\"><math><mi>e</mi><mi>m</mi><mi>f</mi><mo>=</mo><mo>−</mo><mfrac><mrow><mi>Δ</mi><mi>Φ</mi></mrow><mrow><mi>Δ</mi><mi>t</mi></mrow></mfrac></math></span>",
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "2",
    "level": "0",
    "subject_id": "0",
    "topic_id": "105",
    "image": "",
    "difficulty": "متوسط",
    "topic_dbr": "قانون فاراداي للحث الكهرومغناطيسي"
  },
  {
    "id": "PHY-U1-CH3-Q6",
    "question": "<span>في الدينامو (المولد الكهربي)، تُعطى القوة الدافعة الكهربية المستحثة اللحظية (<math><mi>e</mi><mi>m</mi><mi>f</mi></math>) لملف ذي <math><mi>N</mi></math> لفة ومساحة <math><mi>A</mi></math> يدور بسرعة زاوية (<math><mi>ω</mi></math>) في مجال مغناطيسي كثافته (<math><mi>B</mi></math>)، بزاوية <math><mi>θ</mi></math> بين اتجاه السرعة واتجاه الفيض المغناطيسي، بالعلاقة:</span>",
    "choice1": "<span class=\"col-xs-11 ans\"><math><mi>e</mi><mi>m</mi><mi>f</mi><mo>=</mo><mi>N</mi><mi>B</mi><mi>A</mi><mi>ω</mi><mi>cos</mi><mi>θ</mi></math></span>",
    "choice2": "<span class=\"col-xs-11 ans\"><math><mi>e</mi><mi>m</mi><mi>f</mi><mo>=</mo><mi>N</mi><mi>B</mi><mi>A</mi><mi>ω</mi></math></span>",
    "choice3": "<span class=\"col-xs-11 ans\"><math><mi>e</mi><mi>m</mi><mi>f</mi><mo>=</mo><mi>N</mi><mi>B</mi><mi>A</mi><mi>ω</mi><mi>sin</mi><mi>θ</mi></math></span>",
    "choice4": "<span class=\"col-xs-11 ans\"><math><mi>e</mi><mi>m</mi><mi>f</mi><mo>=</mo><mi>B</mi><mi>A</mi><mi>ω</mi><mi>sin</mi><mi>θ</mi></math></span>",
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "3",
    "level": "0",
    "subject_id": "0",
    "topic_id": "106",
    "image": "",
    "difficulty": "متوسط",
    "topic_dbr": "الدينامو والمولد الكهربي"
  },
  {
    "id": "PHY-U1-CH4-Q7",
    "question": "<span>تُعرف المفاعلة السعوية (<math><msub><mi>X</mi><mi>C</mi></msub></math>) بأنها:</span>",
    "choice1": "<span class=\"col-xs-11 ans\">الممانعة التي يلقاها التيار المستمر في المكثف.</span>",
    "choice2": "<span class=\"col-xs-11 ans\">الممانعة التي يلقاها التيار المتردد في المقاومة.</span>",
    "choice3": "<span class=\"col-xs-11 ans\">الممانعة التي يلقاها التيار المتردد في المكثف بسبب سعته.</span>",
    "choice4": "<span class=\"col-xs-11 ans\">الممانعة التي يلقاها التيار المتردد في الملف.</span>",
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "3",
    "level": "0",
    "subject_id": "0",
    "topic_id": "107",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "المفاعلة السعوية"
  },
  {
    "id": "PHY-U2-CH5-Q8",
    "question": "<span>وفقًا لتفسير أينشتاين للظاهرة الكهروضوئية، الشرط الأساسي لتحرير الإلكترونات من سطح معدن عند سقوط فوتونات عليه هو أن تكون:</span>",
    "choice1": "<span class=\"col-xs-11 ans\"><math><mi>h</mi><mi>ν</mi><mo>&lt;</mo><msub><mi>E</mi><mi>W</mi></msub></math></span>",
    "choice2": "<span class=\"col-xs-11 ans\"><math><mi>h</mi><mi>ν</mi><mo>=</mo><mn>0</mn></math></span>",
    "choice3": "<span class=\"col-xs-11 ans\"><math><mi>h</mi><mi>ν</mi><mo>≥</mo><msub><mi>E</mi><mi>W</mi></msub></math></span>",
    "choice4": "<span class=\"col-xs-11 ans\"><math><mi>h</mi><mi>ν</mi><mo>&lt;</mo><mn>0</mn></math></span>",
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "3",
    "level": "0",
    "subject_id": "0",
    "topic_id": "108",
    "image": "",
    "difficulty": "متوسط",
    "topic_dbr": "الظاهرة الكهروضوئية"
  },
  {
    "id": "PHY-U2-CH7-Q9",
    "question": "<span>إحدى الخصائص الأربع الرئيسية لأشعة الليزر هي:</span>",
    "choice1": "<span class=\"col-xs-11 ans\">تشتت الحزمة الضوئية</span>",
    "choice2": "<span class=\"col-xs-11 ans\">تعدد الألوان</span>",
    "choice3": "<span class=\"col-xs-11 ans\">الترابط الفوتوني</span>",
    "choice4": "<span class=\"col-xs-11 ans\">ضعف الشدة</span>",
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "3",
    "level": "0",
    "subject_id": "0",
    "topic_id": "109",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "خصائص الليزر"
  },
  {
    "id": "PHY-U2-CH8-Q10",
    "question": "<span>في بوابة <math><mi>AND</mi></math> المنطقية، متى يكون الخرج بقيمة 1؟</span>",
    "choice1": "<span class=\"col-xs-11 ans\">عندما يكون أي من المدخلات 1</span>",
    "choice2": "<span class=\"col-xs-11 ans\">عندما تكون جميع المدخلات 0</span>",
    "choice3": "<span class=\"col-xs-11 ans\">عندما تكون جميع المدخلات 1</span>",
    "choice4": "<span class=\"col-xs-11 ans\">عندما يكون أحد المدخلات 0</span>",
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "3",
    "level": "0",
    "subject_id": "0",
    "topic_id": "110",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "بوابة AND المنطقية"
  }
]
```



<!-- 


----
----
----
----


- صح وخطآ
---


```json
أنت خبير تقويم تربوي لمادة الفيزياء للمرحلة الثانوية.  \\
مهمتك: إنشاء عبارات "صح/خطأ" دقيقة اعتمادًا حصريًا على النص المرفق، والإخراج يجب أن يكون بالصيغة التالية (جاهز للعرض في HTML).

[متطلبات عامة]  
{"num_questions": 10،
"topic_name": الفصل الاول ،
"topic_id": 2,
"topic_details": اي تفاصيل،
"subject_id": 2,
}
\[مخطط الإخراج النهائي]
[{
"id":"<معرّف السؤال>",
"question":"<span>نص العبارة</span>",
"choice1":"<span class=\"col-xs-11 ans\">صح</span>",
"choice2":"<span class=\"col-xs-11 ans\">خطأ</span>",
"choice3":null,
"choice4":null,
"choice5":null,
"choice6":null,
"choice7":null,
"choice8":null,
"choice9":null,
"choice10":null,
"choice11":null,
"choice12":null,
"correct_answer":"<index الصحيح (1 لصح، 2 لخطأ)>",
"level":"0",
"subject_id":"0",
"topic_id":"<رقم الموضوع>",
"image":"",
"difficulty":"سهل",
"topic_dbr":"<سلسلة الترميز>"},{...},{...}
]

\[إرشادات تصميم المعادلات والدوال الفيزيائية]
1. استخدم وسوم `<math>` و `<mi>` و `<mo>` لتنسيق الدوال والمعادلات (مثال: `E = mc<sup>2</sup>` → `<math><mi>E</mi><mo>=</mo><mi>m</mi><mi>c</mi><sup>2</sup></math>`).
2. استخدم `<sub>` و `<sup>` للرموز السفلية والعلوية.
3. استخدم `<var>` لتحديد الرموز المتغيرة في النص.
4. صيغة جميع الأسئلة متوافقة مع HTML للعرض المباشر في الواجهات.
5. جميع القوالب (MCQ، صح/خطأ، مقالي) تستخدم نفس الهيكل JSON.
6. الأسئلة تُصاغ بالعربية الفصحى وبشكل واضح للطلاب.
\[إرشادات الصياغة]
* 50% عبارات صحيحة، 50% خاطئة.
* صياغة واضحة ومناسبة للطلاب.
* غلّف السؤال والاختيارات بـ `<span>` ليتوافق مع واجهة العرض.
* عند الخطأ → أضف تصحيح مختصر بعد الإجابة.
* correct\_answer هو رقم البديل الصحيح (كنص).
```
---
---
### Examples - True or False 



```json
[
  {
    "id": "PHY-TF-001",
    "question": "<span>ظاهرة التأثير الكهروضوئي تحدث عند سقوط الضوء ذو الشدة العالية على سطح معدني بغض النظر عن تردده.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">صح</span>",
    "choice2": "<span class=\"col-xs-11 ans\">خطأ، بل يتطلب أن يكون تردد الضوء الساقط أكبر من التردد الحرج للمعدن.</span>",
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "2",
    "level": "0",
    "subject_id": "0",
    "topic_id": "5",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "ازدواجية الموجة والجسيم"
  },
  {
    "id": "PHY-TF-002",
    "question": "<span>يتميز ضوء الليزر بالترابط حيث تكون الفوتونات المنبعثة مترابطة زمانياً ومكانياً.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">صح</span>",
    "choice2": "<span class=\"col-xs-11 ans\">خطأ</span>",
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "7",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "الليزر"
  },
  {
    "id": "PHY-TF-003",
    "question": "<span>يتم توصيل مقاومة مجزئ التيار على التوالي مع الجلفانومتر لتحويله إلى أميتر.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">صح</span>",
    "choice2": "<span class=\"col-xs-11 ans\">خطأ، بل يتم توصيل مقاومة مجزئ التيار على التوازي مع الجلفانومتر.</span>",
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "2",
    "level": "0",
    "subject_id": "0",
    "topic_id": "2",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "التأثير المغناطيسي للتيار الكهربي وأجهزة القياس الكهربي"
  },
  {
    "id": "PHY-TF-004",
    "question": "<span>في المحول الكهربي المثالي، تكون الطاقة الكهربية المستهلكة في الملف الابتدائي مساوية للطاقة الكهربية المتولدة في الملف الثانوي.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">صح</span>",
    "choice2": "<span class=\"col-xs-11 ans\">خطأ</span>",
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "3",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "الحث الكهرومغناطيسي"
  },
  {
    "id": "PHY-TF-005",
    "question": "<span>بوابة العاكس (<var>NOT Gate</var>) لها مدخلان ومخرج واحد.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">صح</span>",
    "choice2": "<span class=\"col-xs-11 ans\">خطأ، بل لها مدخل واحد ومخرج واحد.</span>",
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "2",
    "level": "0",
    "subject_id": "0",
    "topic_id": "8",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "الإلكترونيات الحديثة"
  },
  {
    "id": "PHY-TF-006",
    "question": "<span>تتناسب كثافة الفيض المغناطيسي عند مركز ملف دائري يمر به تيار كهربي طردياً مع عدد لفات الملف وعكسياً مع نصف قطره.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">صح</span>",
    "choice2": "<span class=\"col-xs-11 ans\">خطأ</span>",
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "2",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "التأثير المغناطيسي للتيار الكهربي وأجهزة القياس الكهربي"
  },
  {
    "id": "PHY-TF-007",
    "question": "<span>قانون كيرشوف الأول ينص على أن المجموع الجبري للقوى الدافعة الكهربية في دائرة كهربية مغلقة يساوي المجموع الجبري لفروق الجهد في الدائرة.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">صح</span>",
    "choice2": "<span class=\"col-xs-11 ans\">خطأ، بل هذا نص قانون كيرشوف الثاني.</span>",
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "2",
    "level": "0",
    "subject_id": "0",
    "topic_id": "1",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "التيار الكهربي وقانون أوم وقانونا كيرشوف"
  },
  {
    "id": "PHY-TF-008",
    "question": "<span>في أنبوبة كوليدج، يتم إنتاج الأشعة السينية عندما تتصادم الإلكترونات المعجلة مع مادة الهدف.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">صح</span>",
    "choice2": "<span class=\"col-xs-11 ans\">خطأ</span>",
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "6",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "الأطياف الذرية"
  },
  {
    "id": "PHY-TF-009",
    "question": "<span>تنص علاقة دي برولي على أن الطول الموجي المصاحب لجسيم متحرك يتناسب طردياً مع كمية حركة الجسيم.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">صح</span>",
    "choice2": "<span class=\"col-xs-11 ans\">خطأ، بل يتناسب عكسياً مع كمية حركة الجسيم.</span>",
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "2",
    "level": "0",
    "subject_id": "0",
    "topic_id": "5",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "ازدواجية الموجة والجسيم"
  },
  {
    "id": "PHY-TF-010",
    "question": "<span>مولد التيار الكهربي المتردد (الدينامو) هو جهاز يحول الطاقة الميكانيكية إلى طاقة كهربية.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">صح</span>",
    "choice2": "<span class=\"col-xs-11 ans\">خطأ</span>",
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "3",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "الحث الكهرومغناطيسي"
  }
]

```
---
---
---
---

- مقالي 
---


```json
أنت خبير تقويم تربوي لمادة الفيزياء للمرحلة الثانوية.  \\
مهمتك: إنشاء أسئلة مقالية أو إجابة قصيرة اعتمادًا فقط على النص المرفق، والإخراج يجب أن يكون بالصيغة التالية (جاهز للعرض في HTML).

[متطلبات عامة]  
{"num_questions": 10،
"topic_name": الفصل الاول ،
"topic_id": 2,
"topic_details": اي تفاصيل،
"subject_id": 2,
}



\[مخطط الإخراج النهائي]


[
{
"id":"<معرّف السؤال>",
"question":"<span>نص السؤال</span>",
"choice1":"<span class=\"col-xs-11 ans\">إجابة نموذجية مختصرة</span>",
"choice2":null,
"choice3":null,
"choice4":null,
"choice5":null,
"choice6":null,
"choice7":null,
"choice8":null,
"choice9":null,
"choice10":null,
"choice11":null,
"choice12":null,
"correct_answer":"1",
"level":"0",
"subject_id":"0",
"topic_id":"<رقم الموضوع>",
"image":"",
"difficulty":"سهل",
"topic_dbr":"<سلسلة الترميز>"
},
{...},
{...}
]
\[إرشادات تصميم المعادلات والدوال الفيزيائية]

1. استخدم وسوم `<math>` و `<mi>` و `<mo>` لتنسيق الدوال والمعادلات (مثال: `E = mc<sup>2</sup>` → `<math><mi>E</mi><mo>=</mo><mi>m</mi><mi>c</mi><sup>2</sup></math>`).

2. استخدم `<sub>` و `<sup>` للرموز السفلية والعلوية.
3. استخدم `<var>` لتحديد الرموز المتغيرة في النص.
4. صيغة جميع الأسئلة متوافقة مع HTML للعرض المباشر في الواجهات.
5. جميع القوالب (MCQ، صح/خطأ، مقالي) تستخدم نفس الهيكل JSON.
6. الأسئلة تُصاغ بالعربية الفصحى وبشكل واضح للطلاب.


\[إرشادات الصياغة]

* الأسئلة متنوعة (تعريفية، تفسيرية، تطبيقية، تحليلية).
* غلّف السؤال والإجابة بـ `<span>` ليتوافق مع واجهة العرض.
* الإجابة النموذجية في `choice1`.
* باقي الاختيارات null.
```
---
---
### Articles Questions 


```json
[
  {
    "id": "PHY-ESS-001",
    "question": "<span>اذكر نص قانون كيرشوف الأول (قانون حفظ الشحنة الكهربية).</span>",
    "choice1": "<span class=\"col-xs-11 ans\">ينص قانون كيرشوف الأول على أن المجموع الجبري للتيارات الكهربية الداخلة عند نقطة (عقدة) في دائرة كهربية مغلقة يساوي المجموع الجبري للتيارات الخارجة منها (<math><mo>∑</mo><msub><mi>I</mi><mrow><mi>i</mi><mi>n</mi></mrow></msub><mo>=</mo><mo>∑</mo><msub><mi>I</mi><mrow><mi>o</mi><mi>u</mi><mi>t</mi></mrow></msub></math>).</span>",
    "choice2": null,
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "1",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "التيار الكهربي وقانون أوم وقانونا كيرشوف"
  },
  {
    "id": "PHY-ESS-002",
    "question": "<span>على أي عوامل تتوقف المقاومة النوعية لمادة موصل؟</span>",
    "choice1": "<span class=\"col-xs-11 ans\">تتوقف المقاومة النوعية لمادة موصل على نوع المادة ودرجة الحرارة.</span>",
    "choice2": null,
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "1",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "التيار الكهربي وقانون أوم وقانونا كيرشوف"
  },
  {
    "id": "PHY-ESS-003",
    "question": "<span>اشرح كيف يمكن تحويل الجلفانومتر الحساس إلى أميتر.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">يتم تحويل الجلفانومتر إلى أميتر بتوصيل مقاومة صغيرة جداً على التوازي مع ملف الجلفانومتر، تسمى مقاومة مجزئ التيار (<var>R<sub>s</sub></var>).</span>",
    "choice2": null,
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "2",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "التأثير المغناطيسي للتيار الكهربي وأجهزة القياس الكهربي"
  },
  {
    "id": "PHY-ESS-004",
    "question": "<span>اكتب العلاقة التي تعبر عن كفاءة المحول الكهربي.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">كفاءة المحول الكهربي <math><mi>η</mi><mo>=</mo><mfrac><mrow><msub><mi>V</mi><mi>s</mi></msub><mo>×</mo><msub><mi>I</mi><mi>s</mi></msub></mrow><mrow><msub><mi>V</mi><mi>p</mi></msub><mo>×</mo><msub><mi>I</mi><mi>p</mi></msub></mrow></mfrac><mo>=</mo><mfrac><mrow><msub><mi>V</mi><mi>s</mi></msub><mo>×</mo><msub><mi>N</mi><mi>p</mi></msub></mrow><mrow><msub><mi>V</mi><mi>p</mi></msub><mo>×</mo><msub><mi>N</mi><mi>s</mi></msub></mrow></mfrac></math>.</span>",
    "choice2": null,
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "3",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "الحث الكهرومغناطيسي"
  },
  {
    "id": "PHY-ESS-005",
    "question": "<span>ما هي الوظيفة الأساسية لمولد التيار الكهربي المتردد (الدينامو)؟</span>",
    "choice1": "<span class=\"col-xs-11 ans\">هو جهاز يحول الطاقة الميكانيكية إلى طاقة كهربية في مجال مغناطيسي.</span>",
    "choice2": null,
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "3",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "الحث الكهرومغناطيسي"
  },
  {
    "id": "PHY-ESS-006",
    "question": "<span>عرف المفاعلة الحثية واذكر وحدة قياسها.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">المفاعلة الحثية (<var>X<sub>L</sub></var>) هي المعاوقة التي يلقاها التيار المتردد في ملف بسببه الحث الذاتي، ووحدة قياسها هي الأوم (<math><mi>Ω</mi></math>).</span>",
    "choice2": null,
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "4",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "دوائر التيار المتردد"
  },
  {
    "id": "PHY-ESS-007",
    "question": "<span>متى تحدث الظاهرة الكهروضوئية؟</span>",
    "choice1": "<span class=\"col-xs-11 ans\">تحدث الظاهرة الكهروضوئية عند سقوط ضوء على سطح معدني بشرط أن يكون تردد الضوء الساقط أكبر من التردد الحرج للمعدن.</span>",
    "choice2": null,
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "5",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "ازدواجية الموجة والجسيم"
  },
  {
    "id": "PHY-ESS-008",
    "question": "<span>اشرح باختصار كيفية إنتاج الأشعة السينية في أنبوبة كوليدج.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">يتم إنتاج الأشعة السينية في أنبوبة كوليدج بتسخين الفتيلة لتنطلق الإلكترونات، والتي تتصادم مع مادة الهدف بعد تعجيلها بفرق جهد عالٍ، فتنبعث الأشعة السينية نتيجة فقد الإلكترونات لطاقتها.</span>",
    "choice2": null,
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "6",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "الأطياف الذرية"
  },
  {
    "id": "PHY-ESS-009",
    "question": "<span>اذكر ثلاث خصائص مميزة لضوء الليزر.</span>",
    "choice1": "<span class=\"col-xs-11 ans\">من خصائص ضوء الليزر: 1. النقاء الطيفي (Monochromaticity)، 2. الترابط (Coherence)، 3. الشدة العالية (Intensity).</span>",
    "choice2": null,
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "7",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "الليزر"
  },
  {
    "id": "PHY-ESS-010",
    "question": "<span>ما هي وظيفة بوابة العاكس (<var>NOT Gate</var>) في الإلكترونيات الرقمية؟</span>",
    "choice1": "<span class=\"col-xs-11 ans\">تقوم بوابة العاكس (<var>NOT Gate</var>) بعكس حالة المدخل، فإذا كان المدخل <var>HIGH</var> يكون المخرج <var>LOW</var> والعكس صحيح.</span>",
    "choice2": null,
    "choice3": null,
    "choice4": null,
    "choice5": null,
    "choice6": null,
    "choice7": null,
    "choice8": null,
    "choice9": null,
    "choice10": null,
    "choice11": null,
    "choice12": null,
    "correct_answer": "1",
    "level": "0",
    "subject_id": "0",
    "topic_id": "8",
    "image": "",
    "difficulty": "سهل",
    "topic_dbr": "الإلكترونيات الحديثة"
  }
]

```

---
---
---
---



 -->
