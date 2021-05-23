class Tags():
    def __init__(self, data, answer=True):
        question = data['question']
        answer = [ans['text'][0] for ans in data['answers']]
        df = pd.DataFrame(zip(question, answer), columns=['question', 'answer'])
        df['tag'] = None
        
        self.df = df
    
    def who(self):
        kw_who_end1 = ['사람', '인물', '여자', '남자', '이는', '자는', '대상', '상대',
                       '직책', '직업', '직위', '지위', '직책', '신분', '본관', '직분', '계급', '관직']
        kw_who_end2 = ['이는', '자는']
        kw_who_contain = ['누가', '누구', '인물의 이름']
        ner_who = ['CIVILIZATION', 'OCCUPATION']

        idx = self.df['tag'][self.df['tag'].isnull()].index
        for i in idx:
            flag = False
            q = self.df['question'][i]
            q_split = q.split()

            # ~~~~~~~ 사람~~?
            for kw in kw_who_end1:
                if q_split[-1].startswith(kw):
                    self.df['tag'][i] = '[WHO]'
                    flag = True
                    break
            if flag: continue

            # ~~~~~~~ 이는?
            for kw in kw_who_end2:
                if kw == q_split[-1][:-1]:
                    self.df['tag'][i] = '[WHO]'
                    flag = True
                    break
            if flag: continue

            # ~~~~~~~ 감독~~? (NER: CIVILIZATION, OCCUPATION)
            for tag in ner_who:
                if tag == ner(q_split[-1])[0][1]:
                    self.df['tag'][i] = '[WHO]'
                    flag = True
                    break
            if flag: continue

            # ~~~ 누가 ~~~~~~?
            for kw in kw_who_contain:
                if kw in q:
                    self.df['tag'][i] = '[WHO]'
                    flag = True
                    break
            if flag: continue

            # ~~~~~~~ 이름~~?
            if q_split[-1].startswith('이름'):
                # ~~~~~~~ 사람~~ 이름~~?
                for kw in kw_who_end1:
                    if q_split[-2].startswith(kw):
                        self.df['tag'][i] = '[WHO]'
                        flag = True
                        break
                if flag: continue

                # ~~~~~~~ 감독~~ 이름~~? (NER: CIVILIZATION, OCCUPATION)
                for tag in ner_who:
                    if tag == ner(q_split[-2])[0][1]:
                        self.df['tag'][i] = '[WHO]'
                        flag = True
                        break


    def why(self):
        kw_why_end = ['때문']
        kw_why_contain = [' 왜 ', '원인', '요인', '계기']

        idx = self.df['tag'][self.df['tag'].isnull()].index
        for i in idx:
            flag = False
            q = self.df['question'][i]
            q_split = q.split()

            # ~~~ 왜 ~~~~~~?
            for kw in kw_why_contain:
                if kw in q:
                    self.df['tag'][i] = '[WHY]'
                    flag = True
                    break
            if flag: continue

            # ~~~ 때문 ~~~~~~?
            if '때문' in q:
                if (mecab.pos(q_split[-1])[0][1] != 'NNG' or q_split[-1][:2] in ['무엇', '원인', '요인']) and '것은' not in q:
                    self.df['tag'][i] = '[WHY]'
                    continue

            # ~~~ 이유 ~~~~~~?
            if '이유' in q:
                if mecab.pos(q_split[-1])[0][1] != 'NNG' or q_split[-1].startswith('이유'):
                    self.df['tag'][i] = '[WHY]'
                    continue


    def how(self):
        kw_how_contain = ['방법', '어떻게', '어떤 방식', '어떠한 방식']
        kw_how_excptn = ['어떻게 되', '어떻게 돼', '어떻게 된', '방식보다', '방식 보다', '법은']

        idx = self.df['tag'][self.df['tag'].isnull()].index
        for i in idx:
            q = self.df['question'][i]
            q_split = q.split()

            # ~~~ 방법 ~~~~~~?
            for kw in kw_how_contain:
                if kw in q and all([x not in q for x in kw_how_excptn]):
                    self.df['tag'][i] = '[HOW]'
                    break


    def when(self):
        kw_when_end1 = ['년도', '연도', '년대', '연대', '날짜', '요일', '시간', '계절', '시기', '기간', '시대', '얼마만']
        kw_when_end2 = ['해는', '날은', '때는', '달은', '년도는', '연도는']
        kw_when_contain = ['언제', '몇 년', '몇년', '몇 월', '몇월',  '며칠', '몇 시간', '몇시간', '몇 분', '몇분', 
                           '어느 해', '어느 요일', '어느 계절', '어느 시대', '어느 시기', '얼마 뒤', '얼마 후', '얼마 이내', '얼마 이후',
                           '얼마나 걸렸', '얼마나 늦게', '얼마나 빨리', '얼마나 오래', '얼마만에', '얼마동안', '언제부터', '언제까지']

        idx = self.df['tag'][self.df['tag'].isnull()].index
        for i in idx:
            flag = False
            q = self.df['question'][i]
            q_split = q.split()

            # ~~~~~~~ 년도~~?
            for kw in kw_when_end1:
                if q_split[-1].startswith(kw):
                    self.df['tag'][i] = '[WHEN]'
                    flag = True
                    break
            if flag: continue

            # ~~~~~~~ 년도~~?
            for kw in kw_when_end2:
                if q_split[-1][:-1].endswith(kw):
                    self.df['tag'][i] = '[WHEN]'
                    flag = True
                    break
            if flag: continue

            # ~~~~~~~ ~~일은?
            if q_split[-1][:-1].endswith('일은') and len(mecab.pos(q_split[-1])[0][0]) > 1:
                self.df['tag'][i] = '[WHEN]'
                flag = True
                continue

            # ~~~ 언제 ~~~~~~?
            for kw in kw_when_contain:
                if kw in q:
                    self.df['tag'][i] = '[WHEN]'
                    flag = True
                    break
            if flag: continue


    def where(self):
        kw_where_end = ['어디', '곳은', '옮겨', '이동']
        kw_where_contain = ['장소', '위치', '마을', '도시', '나라', '국가', '학교', '출신', '소재지', '근무지', '지역']
        kw_where_excptn = ['사용', '등장']

        idx = self.df['tag'][self.df['tag'].isnull()].index
        for i in idx:
            flag = False
            q = self.df['question'][i]
            q_split = q.split()

            # ~~~~~~~ 장소~~?
            for kw in kw_where_end + kw_where_contain:
                if q_split[-1].startswith(kw):
                    self.df['tag'][i] = '[WHERE]'
                    flag = True
                    break
            if flag: continue

            # ~~~ 장소~~ 이름~~ ~~~~~~?
            nouns = mecab.nouns(q)
            for j in range(len(nouns)-2):
                if nouns[j] in kw_where_contain and '이름' in nouns[j+1:j+3]:
                    self.df['tag'][i] = '[WHERE]'
                    flag = True
                    break
            if flag: continue

            # ~~~ 어느 마을 ~~~~~~?
            for kw in kw_where_contain:
                if '어느 ' + kw in q:
                    self.df['tag'][i] = '[WHERE]'
                    flag = True
                    break
            if flag: continue

            # ~~~ 어디 ~~~~~~?
            if '어디' in q:
                for x in kw_where_excptn:
                    if x not in q:
                        self.df['tag'][i] = '[WHERE]'
                        flag = True
                        break
            if flag: continue

            # ~~~~~~~ ~~지~~?
            for kw in ['지는', '국은', '국가는']:
                if q_split[-1][:-1].endswith(kw) and len(q_split[-1]) > 4:
                    pos = mecab.pos(q_split[-1][:-3])[-1]
                    if pos[1] == 'NNG':
                        self.df['tag'][i] = '[WHERE]'
                        break
    
    
    def quantity(self):
        kw_where_end = ['수는', '양은']
        
        idx = self.df['tag'][self.df['tag'].isnull()].index
        for i in idx:
            q = self.df['question'][i]
            q_split = q.split()
            
            ner_quantity = ner(self.df['answer'][i])
            if len(ner_quantity) == 1 and ner_quantity[0][1] == 'QUANTITY':
                self.df['tag'][i] = '[QUANTITY]'
                continue
            
            if '어느 정도' in q:
                self.df['tag'][i] = '[QUANTITY]'
                continue
                
            if q_split[-1].startswith('얼마') and not q_split[-1].startswith('얼마만'):
                self.df['tag'][i] = '[QUANTITY]'
                continue
            
            for kw in kw_where_end:
                if q_split[-1].startswith(kw):
                    self.df['tag'][i] = '[QUANTITY]'
                    break
            
            
    def cite(self):
        char = ['"', "'", "“", '‘', '《', '≪', '〈', '<', '『', '「', '＜', '《']
        
        idx = self.df['tag'][self.df['tag'].isnull()].index
        for i in idx:
            a = self.df['answer'][i]
            for c in char:
                if a[0] == c:
                    self.df['tag'][i] = '[CITE]'
                    break

                
    def apply(self):
        self.who()
        self.why()
        self.how()
        self.when()
        self.where()
        self.quantity()
        self.cite()
        self.df['tag'] = self.df['tag'].apply(lambda x: '[WHAT]' if x is None else x)
        
        return self.df