from evaluate import qa_result


CONTEXT = 'The COVID‑19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 (COVID‑19), caused by severe acute respiratory syndrome coronavirus 2 (SARS‑CoV‑2). The outbreak was first identified in December 2019 in Wuhan, China. The World Health Organization declared the outbreak a Public Health Emergency of International Concern on 30 January 2020 and a pandemic on 11 March. As of 24 August 2020, more than 23.4 million cases of COVID‑19 have been reported in more than 188 countries and territories, resulting in more than 808,000 deaths; more than 15.1 million people have recovered.'
QUERIES = ['where was the outbreak first identified ?',
           'when was the outbreak first identified ?',
           'how many people have died from covid-19 ?',
           'how many people have recovered from covid-19 ?',
           'when did the world health organization declare an emergency ?',
           'when did the world health organization declare a pandemic ?']

for query in QUERIES:
    print(qa_result(CONTEXT, query))