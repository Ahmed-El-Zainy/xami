### Prompts

---

- اختيارات

---

- Include both Maths and Normal Questions

```json

أنت خبير تقويم تربوي لمادة الفيزياء للمرحلة الثانوية.
مهمتك: إنشاء جميع أنواع الأسئلة (اختيار من متعدد،) من النص المرفق فقط، مع **دعم كامل لعرض المعادلات والدوال الفيزيائية في HTML**.
\[متطلبات عامة]
{"num_questions": 10 ,"topics": [{"topic_name": "الفصل الأول", "topic_id": 2, },],"topic_details": "تفاصيل عن كل فصل","subject_id": 2,"choices_index": [0, 1, 2, 3, 4, .......], "difficulty": [1%:100%]}
\[مخطط الإخراج النهائي]
[{"id": "<معرّف السؤال>","question": "<span>نص السؤال (استخدم وسوم HTML لعرض الدوال والمعادلات الفيزيائية، مثل <math><mi>F</mi><mo>=</mo><mi>m</mi><mo>×</mo><mi>a</mi></math>)</span>",
  "choice1": "<span class=\"col-xs-11 ans\">الاختيار الأول</span>",
  "choice2": "<span class=\"col-xs-11 ans\">الاختيار الثاني</span>",
  "choice3": "<span class=\"col-xs-11 ans\">الاختيار الثالث</span>",
  "choice4": "<span class=\"col-xs-11 ans\">الاختيار الرابع</span>",
  "correct_answer": "<index الصحيح>","level": "0","subject_id": "0","topic_id": "<رقم الموضوع>","difficulty": "1","topic_dbr": "<سلسلة الترميز>"}]
\[إرشادات تصميم المعادلات والدوال الفيزيائية]
1.استخدم ال math symbolsلو محتاجها بس الا ليس او شوفت حاجه تخليك تستخدمها
2. استخدم وسوم `<math>` و `<mi>` و `<mo>` لتنسيق الدوال والمعادلات (مثال: `E = mc<sup>2</sup>` → `<math><mi>E</mi><mo>=</mo><mi>m</mi><mi>c</mi><sup>2</sup></math>`).
3. استخدم `<var>` لتحديد الرموز المتغيرة. 7. استخدم `<math>` أو `<sup>/<sub>` للمعادلات.  استخد`<sub>` و `<sup>` للرموز السفلية والعلوية.
[إرشادات توليد الأسئلة]
5. كل سؤال له 4 بدائل، بديل واحد صحيح. خلي الموديل يقيم صعوبه الاسئلة من 1  لحد 100 
6. غلّف السؤال بـ `<span>` والاختيارات بـ `<span class=\"col-xs-11ans\">…</span>`."
7. `correct_answer` هو رقم البديل الصحيح كنص . لا تستخدم "كل ما سبق" إلا لو ورد بالنص.
8. وزّع المستويات (سهل/متوسط/متقدم).  لا تضف أي معلومة خارج النص.
9. لا تكرر أي إجابة صحيحة أو تعطي أكثر من إجابة صحيحة. لا تضع أي توضيح للإجابات في المخرجات.

```

---

- Tokens: 678
- Characters: 1885

---

---

---

---

- معادلات وحسابات فيزيائية

```json

أنت خبير تقويم تربوي لمادة الفيزياء للمرحلة الثانوية.
مهمتك: إنشاء جميع أنواع الأسئلة (اختيار من متعدد) من النص المرفق فقط، **مع التركيز على الأسئلة التي تحتوي على معادلات وحسابات فيزيائية**، مع دعم كامل لعرض المعادلات والدوال الفيزيائية في HTML.
[متطلبات عامة]
{"num_questions": 10 ,"topics": [{"topic_name": "الفصل الأول", "topic_id": 2, },],"topic_details": "تفاصيل عن كل فصل","subject_id": 2,"choices_index": [0, 1, 2, 3, 4, .......], "difficulty":[1%:100%]}
[مخطط الإخراج النهائي]
[{"id": "<معرّف السؤال>",
  "question": "<span>نص السؤال (استخدم وسوم HTML لعرض المعادلات الفيزيائية، مثل <math><mi>F</mi><mo>=</mo><mi>m</mi><mo>×</mo><mi>a</mi></math>)</span>",
  "choice1": "<span class=\"col-xs-11 ans\">الاختيار الأول</span>",
  "choice2": "<span class=\"col-xs-11 ans\">الاختيار الثاني</span>",
  "choice3": "<span class=\"col-xs-11 ans\">الاختيار الثالث</span>",
  "choice4": "<span class=\"col-xs-11 ans\">الاختيار الرابع</span>",
  "correct_answer": "<index الصحيح>",
  "level": "0","subject_id": "0","topic_id": "<رقم الموضوع>","difficulty": "1","topic_dbr": "<سلسلة الترميز>"}]
[إرشادات تصميم المعادلات]
1. ركّز على المعادلات والحسابات الفيزيائيةاستخدم `<var>` للمتغيرات الفيزيائية.
2. استخدم `<math>`, `<mi>`, `<mo>`، `<sup>`, `<sub>` لتنسيق المعادلات.
[إرشادات توليد الأسئلة]
3. كل سؤال له 4 بدائل، بديل واحد صحيح. خلي الموديل يقيم صعوبه الاسئلة من 1  لحد 100 
4. غلّف السؤال بـ `<span>` والاختيارات بـ `<span class="col-xs-11 ans">…</span>`.
5. وزّع المستويات (سهل/متوسط/متقدم). correct_answer رقم البديل الصحيح كنص.
6. لا تضف أي معلومة غير موجودة بالنص.
7. لا تضف أي شرح للإجابات.,لا تكرر الإجابة الصحيحة أو تضيف أكثر من إجابة صحيحة.
```

---

- Tokens: 565
- Characters: 1615

---

---

---

---

- لنظرية والتعريفية والمفاهيم الأساسي

```json

أنت خبير تقويم تربوي لمادة الفيزياء للمرحلة الثانوية.
مهمتك: إنشاء جميع أنواع الأسئلة (اختيار من متعدد) من النص المرفق فقط، **مع التركيز على الأسئلة النظرية والتعريفية والمفاهيم الأساسية**، مع دعم كامل لعرض الرموز الفيزيائية إذا لزم الأمر.
[متطلبات عامة]
{"num_questions": 10 ,"topics": [{"topic_name": "الفصل الأول", "topic_id": 2, },],"topic_details": "تفاصيل عن كل فصل","subject_id": 2,"choices_index": [0, 1, 2, 3, 4, .......], "difficulty":[1%:100%]}
[مخطط الإخراج النهائي]
[{"id": "<معرّف السؤال>",
  "question": "<span>نص السؤال (استخدم HTML لتنسيق الرموز إذا لزم، مثل <var>g</var> لتسارع الجاذبية)</span>",
  "choice1": "<span class=\"col-xs-11 ans\">الاختيار الأول</span>",
  "choice2": "<span class=\"col-xs-11 ans\">الاختيار الثاني</span>",
  "choice3": "<span class=\"col-xs-11 ans\">الاختيار الثالث</span>",
  "choice4": "<span class=\"col-xs-11 ans\">الاختيار الرابع</span>",
  "correct_answer": "<index الصحيح>",
  "level": "0","subject_id": "0","topic_id": "<رقم الموضوع>","difficulty": "1","topic_dbr": "<سلسلة الترميز>"}]
[إرشادات توليد الأسئلة]
1. ركّز على التعريفات، المفاهيم، النظريات.  لا تضف أي معلومة غير موجودة بالنص.
2. كل سؤال له 4 بدائل، بديل واحد صحيح. خلي الموديل يقيم صعوبه الاسئلة من 1  لحد 100 
3. غلّف السؤال بـ `<span>` والاختيارات بـ `<span class="col-xs-11 ans">…</span>`.
4. وزّع المستويات (سهل/متوسط/متقدم). correct_answer رقم البديل الصحيح كنص.
5. لا تضف أي شرح للإجابات.,لا تكرر الإجابة الصحيحة أو تضيف أكثر من إجابة صحيحة.
```

---

- Tokens: 502
- Characters: 1464

---

---

<!-- ### Examples Multiple Choice

---
أهلاً بك! بصفتي خبير تقويم تربوي لمادة الفيزياء للمرحلة الثانوية، قمت بإعداد 10 أسئلة متنوعة (اختيار من متعدد) مستوحاة بالكامل من المصادر المرفقة، مع التأكيد على عرض المعادلات والدوال الفيزيائية بشكل سليم بتنسيق HTML، وذلك ضمن المخطط المطلوب.

إليك الأسئلة:

```json
// أهلاً بك أيها الخبير! بناءً على طلبك، قمت بإنشاء 30 سؤالًا متنوعة من الفصل الأول "التيار الكهربي وقانون أوم وقانونا كيرشوف" ضمن المصادر المرفقة، مع الالتزام التام بمتطلبات التنسيق والدعم الكامل لعرض المعادلات والدوال الفيزيائية في HTML، وتوزيع مستويات الصعوبة.

// إليك الأسئلة بتنسيق JSON المطلوب:

// ```json
// [
//   {
//     "id": "1",
//     "question": "<span dir=\"rtl\">ما هي العلاقة الرياضية لشدة التيار الكهربي <math><var>I</var></math>، إذا مرت كمية كهربية <math><var>Q</var></math> خلال مقطع موصل في زمن <math><var>t</var></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>I</var><mo>=</mo><var>Q</var><mo>⋅</mo><var>t</var></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>I</var><mo>=</mo><mfrac><var>t</var><var>Q</var></mfrac></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>I</var><mo>=</mo><mfrac><var>Q</var><var>t</var></mfrac></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>I</var><mo>=</mo><mfrac><mn>1</mn><mrow><var>Q</var><mo>⋅</mo><var>t</var></mrow></mfrac></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "2",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_current_intensity_formula"
//   },
//   {
//     "id": "2",
//     "question": "<span dir=\"rtl\">ما هي العلاقة الرياضية لفرق الجهد <math><var>V</var></math>، إذا كان الشغل المبذول لنقل كمية من الكهربية <math><var>Q</var></math> هو <math><var>W</var></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>V</var><mo>=</mo><var>W</var><mo>⋅</mo><var>Q</var></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>V</var><mo>=</mo><mfrac><var>Q</var><var>W</var></mfrac></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>V</var><mo>=</mo><mfrac><var>W</var><var>Q</var></mfrac></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>V</var><mo>=</mo><mfrac><mn>1</mn><mrow><var>W</var><mo>⋅</mo><var>Q</var></mrow></mfrac></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "2",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_potential_difference_formula"
//   },
//   {
//     "id": "3",
//     "question": "<span dir=\"rtl\">أي من العلاقات التالية تمثل قانون أوم لفرق الجهد <math><var>V</var></math> والتيار <math><var>I</var></math> والمقاومة <math><var>R</var></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>R</var><mo>=</mo><var>V</var><mo>⋅</mo><var>I</var></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>I</var><mo>=</mo><var>V</var><mo>⋅</mo><var>R</var></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>V</var><mo>=</mo><mfrac><var>R</var><var>I</var></mfrac></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>R</var><mo>=</mo><mfrac><var>V</var><var>I</var></mfrac></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "3",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_ohm_law_formula"
//   },
//   {
//     "id": "4",
//     "question": "<span dir=\"rtl\">ما هي وحدة قياس المقاومة النوعية <math><mi>ρ</mi></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mi>Ω</mi></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mi>Ω</mi><mo>.</mo><mi>m</mi></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><msup><mi>Ω</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup><mo>.</mo><msup><mi>m</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mi>A</mi></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "1",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_resistivity_unit"
//   },
//   {
//     "id": "5",
//     "question": "<span dir=\"rtl\">ما هي العلاقة الرياضية الصحيحة التي تربط التوصيلية الكهربية <math><mi>σ</mi></math> بالمقاومة النوعية <math><mi>ρ</mi></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mi>σ</mi><mo>=</mo><mi>ρ</mi></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mi>σ</mi><mo>=</mo><mfrac><mn>1</mn><mi>ρ</mi></mfrac></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mi>σ</mi><mo>=</mo><msup><mi>ρ</mi><mn>2</mn></msup></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mi>σ</mi><mo>=</mo><mfrac><mi>ρ</mi><mn>2</mn></mfrac></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "1",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_conductivity_formula"
//   },
//   {
//     "id": "6",
//     "question": "<span dir=\"rtl\">عند توصيل ثلاث مقاومات <math><msub><var>R</var><mn>1</mn></msub><mo>,</mo><msub><var>R</var><mn>2</mn></msub><mo>,</mo><msub><var>R</var><mn>3</mn></msub></math> على التوالي، ما هي المقاومة المكافئة <math><msub><var>R</var><mrow><mi>eq</mi></mrow></msub></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><msub><var>R</var><mrow><mi>eq</mi></mrow></msub><mo>=</mo><msub><var>R</var><mn>1</mn></msub><mo>+</mo><msub><var>R</var><mn>2</mn></msub><mo>+</mo><msub><var>R</var><mn>3</mn></msub></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mfrac><mn>1</mn><msub><var>R</var><mrow><mi>eq</mi></mrow></msub></mfrac><mo>=</mo><mfrac><mn>1</mn><msub><var>R</var><mn>1</mn></msub></mfrac><mo>+</mo><mfrac><mn>1</mn><msub><var>R</var><mn>2</mn></msub></mfrac><mo>+</mo><mfrac><mn>1</mn><msub><var>R</var><mn>3</mn></msub></mfrac></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><msub><var>R</var><mrow><mi>eq</mi></mrow></msub><mo>=</mo><mfrac><mrow><msub><var>R</var><mn>1</mn></msub><msub><var>R</var><mn>2</mn></msub><msub><var>R</var><mn>3</mn></msub></mrow><mrow><msub><var>R</var><mn>1</mn></msub><mo>+</mo><msub><var>R</var><mn>2</mn></msub><mo>+</mo><msub><var>R</var><mn>3</mn></msub></mrow></mfrac></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><msub><var>R</var><mrow><mi>eq</mi></mrow></msub><mo>=</mo><msub><var>R</var><mn>1</mn></msub><mo>⋅</mo><msub><var>R</var><mn>2</mn></msub><mo>⋅</mo><msub><var>R</var><mn>3</mn></msub></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "0",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_series_resistance_formula"
//   },
//   {
//     "id": "7",
//     "question": "<span dir=\"rtl\">عند توصيل ثلاث مقاومات <math><msub><var>R</var><mn>1</mn></msub><mo>,</mo><msub><var>R</var><mn>2</mn></msub><mo>,</mo><msub><var>R</var><mn>3</mn></msub></math> على التوازي، ما هي المقاومة المكافئة <math><msub><var>R</var><mrow><mi>eq</mi></mrow></msub></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><msub><var>R</var><mrow><mi>eq</mi></mrow></msub><mo>=</mo><msub><var>R</var><mn>1</mn></msub><mo>+</mo><msub><var>R</var><mn>2</mn></msub><mo>+</mo><msub><var>R</var><mn>3</mn></msub></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mfrac><mn>1</mn><msub><var>R</var><mrow><mi>eq</mi></mrow></msub></mfrac><mo>=</mo><mfrac><mn>1</mn><msub><var>R</var><mn>1</mn></msub></mfrac><mo>+</mo><mfrac><mn>1</mn><msub><var>R</var><mn>2</mn></msub></mfrac><mo>+</mo><mfrac><mn>1</mn><msub><var>R</var><mn>3</mn></msub></mfrac></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><msub><var>R</var><mrow><mi>eq</mi></mrow></msub><mo>=</mo><mfrac><mrow><msub><var>R</var><mn>1</mn></msub><msub><var>R</var><mn>2</mn></msub><msub><var>R</var><mn>3</mn></msub></mrow><mrow><msub><var>R</var><mn>1</mn></msub><mo>+</mo><msub><var>R</var><mn>2</mn></msub><mo>+</mo><msub><var>R</var><mn>3</mn></msub></mrow></mfrac></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><msub><var>R</var><mrow><mi>eq</mi></mrow></msub><mo>=</mo><msub><var>R</var><mn>1</mn></msub><mo>⋅</mo><msub><var>R</var><mn>2</mn></msub><mo>⋅</mo><msub><var>R</var><mn>3</mn></msub></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "1",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_parallel_resistance_formula"
//   },
//   {
//     "id": "8",
//     "question": "<span dir=\"rtl\">أي من العلاقات التالية تمثل قانون أوم للدائرة المغلقة لشدة التيار <math><var>I</var></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>I</var><mo>=</mo><frac><var>V</var><var>R</var></mfrac></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>I</var><mo>=</mo><msub><var>V</var><mi>B</mi></msub><mo>+</mo><var>R</var><mo>+</mo><var>r</var></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>I</var><mo>=</mo><frac><msub><var>V</var><mi>B</mi></msub><mrow><var>R</var><mo>+</mo><var>r</var></mrow></mfrac></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><var>I</var><mo>=</mo><msub><var>V</var><mi>B</mi></msub><mo>⋅</mo><mrow><mo>(</mo><var>R</var><mo>+</mo><var>r</var><mo>)</mo></mrow></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "2",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "متوسط",
//     "topic_dbr": "physics_chapter1_ohm_law_closed_circuit_formula"
//   },
//   {
//     "id": "9",
//     "question": "<span dir=\"rtl\">ما هو المبدأ الذي ينص عليه قانون كيرشوف الأول؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">المجموع الجبري للتيارات الكهربية عند نقطة تفرع يساوي صفرًا.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">مجموع القوى الدافعة الكهربية في أي مسار مغلق يساوي صفرًا.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">مجموع فروق الجهد في أي مسار مغلق يساوي مجموع القوى الدافعة الكهربية.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">شدة التيار تتناسب طرديًا مع فرق الجهد عند ثبوت درجة الحرارة.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "0",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_kirchhoffs_first_law_statement"
//   },
//   {
//     "id": "10",
//     "question": "<span dir=\"rtl\">ما هو المبدأ الذي ينص عليه قانون كيرشوف الثاني؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">المجموع الجبري للتيارات الكهربية عند نقطة تفرع يساوي صفرًا.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">المجموع الجبري للقوى الدافعة الكهربية في دائرة كهربية مغلقة يساوي المجموع الجبري لفروق الجهد في الدائرة.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">شدة التيار المار في موصل تتناسب عكسيًا مع مقاومة الموصل.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">التوصيلية الكهربية للمادة تتناسب طرديًا مع درجة حرارتها.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "1",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_kirchhoffs_second_law_statement"
//   },
//   {
//     "id": "11",
//     "question": "<span dir=\"rtl\">وصلت ثلاث مقاومات <math><mn>25</mn><mi>Ω</mi></math> و <math><mn>70</mn><mi>Ω</mi></math> و <math><mn>85</mn><mi>Ω</mi></math> على التوالي مع بطارية قوتها الدافعة الكهربية <math><mn>45</mn><mi>V</mi></math> (بإهمال المقاومة الداخلية). احسب شدة التيار الكلي المار في الدائرة.</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>180</mn><mi>A</mi></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>0.25</mn><mi>A</mi></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>45</mn><mi>A</mi></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>1.8</mn><mi>A</mi></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "1",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "متوسط",
//     "topic_dbr": "physics_chapter1_series_current_calculation"
//   },
//   {
//     "id": "12",
//     "question": "<span dir=\"rtl\">إذا وصلت ثلاث مقاومات <math><mn>25</mn><mi>Ω</mi></math> و <math><mn>70</mn><mi>Ω</mi></math> و <math><mn>85</mn><mi>Ω</mi></math> على التوازي مع مصدر فرق جهده <math><mn>45</mn><mi>V</mi></math> (بإهمال المقاومة الداخلية)، ما هي شدة التيار المار في المقاومة <math><mn>25</mn><mi>Ω</mi></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>0.643</mn><mi>A</mi></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>1.8</mn><mi>A</mi></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>0.529</mn><mi>A</mi></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>2.972</mn><mi>A</mi></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "1",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "متوسط",
//     "topic_dbr": "physics_chapter1_parallel_current_calculation"
//   },
//   {
//     "id": "13",
//     "question": "<span dir=\"rtl\">ما هي المقاومة المكافئة لمقاومتين <math><msub><var>R</var><mn>1</mn></msub></math> و <math><msub><var>R</var><mn>2</mn></msub></math> متصلتين على التوازي؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><msub><var>R</var><mrow><mi>eq</mi></mrow></msub><mo>=</mo><msub><var>R</var><mn>1</mn></msub><mo>+</mo><msub><var>R</var><mn>2</mn></msub></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><msub><var>R</var><mrow><mi>eq</mi></mrow></msub><mo>=</mo><mfrac><mrow><msub><var>R</var><mn>1</mn></msub><mo>+</mo><msub><var>R</var><mn>2</mn></msub></mrow><mrow><msub><var>R</var><mn>1</mn></msub><msub><var>R</var><mn>2</mn></msub></mrow></mfrac></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><msub><var>R</var><mrow><mi>eq</mi></mrow></msub><mo>=</mo><mfrac><mrow><msub><var>R</var><mn>1</mn></msub><msub><var>R</var><mn>2</mn></msub></mrow><mrow><msub><var>R</var><mn>1</mn></msub><mo>+</mo><msub><var>R</var><mn>2</mn></msub></mrow></mfrac></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><msub><var>R</var><mrow><mi>eq</mi></mrow></msub><mo>=</mo><mfrac><mn>1</mn><msub><var>R</var><mn>1</mn></msub></mfrac><mo>+</mo><mfrac><mn>1</mn><msub><var>R</var><mn>2</mn></msub></mfrac></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "2",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "متوسط",
//     "topic_dbr": "physics_chapter1_two_parallel_resistors_formula"
//   },
//   {
//     "id": "14",
//     "question": "<span dir=\"rtl\">في دائرة كهربية مغلقة، ما هو الاتجاه التقليدي للتيار الكهربي خارج المصدر؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">من القطب السالب إلى القطب الموجب.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">من القطب الموجب إلى القطب السالب.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">عكس اتجاه حركة الإلكترونات.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">التيار يتغير اتجاهه باستمرار.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "1",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_conventional_current_direction"
//   },
//   {
//     "id": "15",
//     "question": "<span dir=\"rtl\">أي من العوامل التالية لا تعتمد عليها مقاومة موصل <math><var>R</var></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">نوع المادة المصنوع منها الموصل.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">طول الموصل.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">مساحة مقطع الموصل.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">شدة التيار المار فيه.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "3",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_factors_affecting_resistance"
//   },
//   {
//     "id": "16",
//     "question": "<span dir=\"rtl\">وصلت أربع مقاومات متماثلة كل منها <math><mn>6</mn><mi>Ω</mi></math> على التوازي مع بطارية <math><mn>12</mn><mi>V</mi></math> ذات مقاومة داخلية مهملة. ما هي المقاومة الكلية لهذه المقاومات الأربع؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>24</mn><mi>Ω</mi></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>12</mn><mi>Ω</mi></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>1.5</mn><mi>Ω</mi></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>6</mn><mi>Ω</mi></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "2",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "متوسط",
//     "topic_dbr": "physics_chapter1_parallel_identical_resistors_calculation"
//   },
//   {
//     "id": "17",
//     "question": "<span dir=\"rtl\">ما هو تعريف الفولت (<var>Volt</var>)؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">وحدة قياس شدة التيار الكهربي.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">وحدة قياس الطاقة الكهربية.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">فرق الجهد بين نقطتين عندما يلزم شغل قدره <math><mn>1</mn><mi>Joule</mi></math> لنقل شحنة <math><mn>1</mn><mi>Coulomb</mi></math> بينهما.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">وحدة قياس المقاومة الكهربية.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "2",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "متوسط",
//     "topic_dbr": "physics_chapter1_definition_volt"
//   },
//   {
//     "id": "18",
//     "question": "<span dir=\"rtl\">كيف يوصل الأميتر في الدوائر الكهربية لقياس شدة التيار؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">على التوازي.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">على التوالي.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">بأي طريقة.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">لا يوصل.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "1",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_ammeter_connection"
//   },
//   {
//     "id": "19",
//     "question": "<span dir=\"rtl\">كيف يوصل الفولتميتر في الدوائر الكهربية لقياس فرق الجهد؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">على التوالي.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">على التوازي.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">بأي طريقة.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">لا يوصل.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "1",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_voltmeter_connection"
//   },
//   {
//     "id": "20",
//     "question": "<span dir=\"rtl\">ينص قانون كيرشوف الأول على أن مجموع التيارات الداخلة إلى نقطة يساوي مجموع التيارات الخارجة منها.</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">صحيح</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">خطأ</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">غير صحيح دائمًا</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">يتوقف على نوع الدائرة</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "0",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_kirchhoff_first_law_true_false"
//   },
//   {
//     "id": "21",
//     "question": "<span dir=\"rtl\">المجموع الجبري للقوى الدافعة الكهربية في أي مسار مغلق يساوي المجموع الجبري لفروق الجهد <math><mo>∑</mo><msub><var>V</var><mi>B</mi></msub><mo>=</mo><mo>∑</mo><var>I</var><mo>.</mo><var>R</var></math>.</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">صحيح</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">خطأ</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">غير صحيح دائمًا</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">يتوقف على اتجاه التيار</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "0",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_kirchhoff_second_law_true_false"
//   },
//   {
//     "id": "22",
//     "question": "<span dir=\"rtl\">ما هو تعريف المقاومة <math><var>R</var></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">كمية الشغل المبذول لنقل شحنة كهربائية.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">النسبة بين فرق الجهد إلى شدة التيار.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">معدل تدفق الشحنات الكهربائية.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">مقدرة الموصل على تمرير التيار الكهربائي.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "1",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "متوسط",
//     "topic_dbr": "physics_chapter1_resistance_definition"
//   },
//   {
//     "id": "23",
//     "question": "<span dir=\"rtl\">طبقاً لقانون أوم، إذا زادت شدة التيار <math><var>I</var></math> في دائرة مع ثبوت المقاومة <math><var>R</var></math>، فماذا يحدث لفرق الجهد <math><var>V</var></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">يزداد.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">يقل.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">يبقى ثابتًا.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">لا يمكن التنبؤ به.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "0",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "متوسط",
//     "topic_dbr": "physics_chapter1_ohm_law_relationship"
//   },
//   {
//     "id": "24",
//     "question": "<span dir=\"rtl\">في دائرة التوالي، إذا كان لديك عدة مقاومات، كيف يتوزع فرق الجهد الكلي عبر هذه المقاومات؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">يتساوى فرق الجهد عبر كل مقاومة.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">يتناسب فرق الجهد عكسيًا مع قيمة المقاومة.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">يتجزأ فرق الجهد الكلي بحيث يكون مجموع فروق الجهد على المقاومات مساويًا لفرق الجهد الكلي.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">فرق الجهد الكلي يتركز على المقاومة الأكبر.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "2",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "متوسط",
//     "topic_dbr": "physics_chapter1_voltage_distribution_series"
//   },
//   {
//     "id": "25",
//     "question": "<span dir=\"rtl\">في دائرة التوازي، إذا كان لديك عدة مقاومات، كيف يتوزع التيار الكلي المار في هذه المقاومات؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">يتجزأ التيار الكلي بحيث يمر الجزء الأكبر في المقاومة الأصغر.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">يتساوى التيار المار في كل مقاومة.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">يتناسب التيار طرديًا مع قيمة المقاومة.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">التيار الكلي يتركز في المقاومة الأكبر.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "0",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "متوسط",
//     "topic_dbr": "physics_chapter1_current_distribution_parallel"
//   },
//   {
//     "id": "26",
//     "question": "<span dir=\"rtl\">في قانون أوم للدائرة المغلقة، ماذا يمثل الرمز <math><var>r</var></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">المقاومة الخارجية.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">المقاومة الداخلية للمصدر.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">شدة التيار.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">القوة الدافعة الكهربية.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "1",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_internal_resistance_symbol"
//   },
//   {
//     "id": "27",
//     "question": "<span dir=\"rtl\">في الدائرة الموضحة بالشكل (1-16) بصفحة 20، التي تحتوي على ثلاث بطاريات (<math><mn>20</mn><mi>V</mi></math>, <math><mn>30</mn><mi>V</mi></math>, <math><mn>15</mn><mi>V</mi></math>) وثلاث مقاومات (<math><mn>1</mn><mi>Ω</mi></math>, <math><mn>2</mn><mi>Ω</mi></math>, <math><mn>5</mn><mi>Ω</mi></math>)، وبعد تطبيق قوانين كيرشوف، ما هي قيمة شدة التيار <math><msub><var>I</var><mn>1</mn></msub></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>1.46</mn><mi>A</mi></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mo>-</mo><mn>2.35</mn><mi>A</mi></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>3.82</mn><mi>A</mi></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>0.5</mn><mi>A</mi></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "1",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "متقدم",
//     "topic_dbr": "physics_chapter1_kirchhoffs_complex_calculation_I1"
//   },
//   {
//     "id": "28",
//     "question": "<span dir=\"rtl\">في الدائرة الموضحة بالشكل (1-22) بصفحة 27، ما هو فرق الجهد بين النقطتين <math><var>b</var></math> و <math><var>c</var></math>؟</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>15</mn><mi>V</mi></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>5</mn><mi>V</mi></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>8</mn><mi>V</mi></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>10</mn><mi>V</mi></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "2",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "متقدم",
//     "topic_dbr": "physics_chapter1_kirchhoffs_voltage_two_batteries_Vbc"
//   },
//   {
//     "id": "29",
//     "question": "<span dir=\"rtl\">عندما يمر تيار كهربي شدته <math><mn>3</mn><mi>A</mi></math> عبر نقطة في دائرة كهربية، فإن الشحنة الكهربية <math><var>Q</var></math> التي تمر خلال دقيقة واحدة (<math><mn>60</mn></math> ثانية) تساوي:</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>3</mn><mi>C</mi></math></span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>60</mn><mi>C</mi></math></span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>180</mn><mi>C</mi></math></span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\"><math><mn>0.05</mn><mi>C</mi></math></span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "2",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_charge_calculation"
//   },
//   {
//     "id": "30",
//     "question": "<span dir=\"rtl\">لكي يطبق قانون أوم على موصل، يجب أن يكون فرق الجهد بين طرفيه متناسبًا طرديًا مع شدة التيار، بشرط:</span>",
//     "choice1": "<span class=\"col-xs-11 ans\" dir=\"rtl\">تغير درجة الحرارة.</span>",
//     "choice2": "<span class=\"col-xs-11 ans\" dir=\"rtl\">زيادة المقاومة.</span>",
//     "choice3": "<span class=\"col-xs-11 ans\" dir=\"rtl\">ثبوت درجة الحرارة.</span>",
//     "choice4": "<span class=\"col-xs-11 ans\" dir=\"rtl\">زيادة شدة التيار.</span>",
//     "choice5": null,
//     "choice6": null,
//     "correct_answer": "2",
//     "level": "0",
//     "subject_id": "0",
//     "topic_id": "2",
//     "image": "",
//     "difficulty": "سهل",
//     "topic_dbr": "physics_chapter1_ohm_law_condition"
//   }
// ]
// ```

-->
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
