import json
import pandas as pd
import requests


def get_problems(handle: str) -> pd.DataFrame:
    url = f'https://solved.ac/api/v3/search/problem?query=%40{handle}&sort=id&direction=asc'
    i = 1

    data = {'id': [], 'title': [], 'tags': [], 'level': []}

    while True:
        data_dict = requests.get(url + f"&page={i}").json()

        # 문제 목록을 저장
        problems = data_dict['items']

        if problems == []:
            break

        # 각 문제의 제목과 태그 추출
        for problem in problems:
            id = problem['problemId']

            title = problem['titleKo']

            tags_raw = problem['tags']
            tags = [tag['key'] for tag in tags_raw]

            level = problem['level']

            data['id'].append(id)
            data['title'].append(title)
            data['tags'].append(tags)
            data['level'].append(level)

        i += 1

    df = pd.DataFrame(data)
    return df


df = get_problems("hoseong8115")
print(df)
df.to_csv("solvedac.csv", index=False)
df[:50].to_csv("solvedac_cut.csv", index=False)