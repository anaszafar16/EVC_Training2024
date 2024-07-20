import requests
from bs4 import BeautifulSoup
import csv

date = input('Enter date (YYYY-MM-DD): ')
page = requests.get(f'https://www.yallakora.com/match-center/?date={date}#')

def main(page):
    src = page.content
    soup = BeautifulSoup(src, "lxml")
    matches_details = []

    championships = soup.find_all("div", {'class': 'matchCard'})

    def get_match_info(championship):
        championship_title = championship.find('a', {'class': 'tourTitle'}).text.strip()
        all_matches = championship.find_all('div', {'class': 'item'})

        for match in all_matches:
            team_a = match.find('div', {'class': 'teamA'}).text.strip()
            team_b = match.find('div', {'class': 'teamB'}).text.strip()

            match_result = match.find('div', {'class': 'MResult'}).find_all('span', {'class': 'score'})

            score = f"{match_result[0].text.strip()} - {match_result[1].text.strip()}"

            match_time = match.find('div', {'class': 'MResult'}).find("span", {'class': 'time'}).text.strip()

            matches_details.append({
                "championship_title": championship_title,
                "team_a": team_a,
                "team_b": team_b,
                "score": score,
                "match_time": match_time
            })

    for championship in championships:
        get_match_info(championship)

    if matches_details:
        keys = matches_details[0].keys()
        
        with open('matches.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(matches_details)
    else:
        print("No matches found for the given date.")

main(page)
