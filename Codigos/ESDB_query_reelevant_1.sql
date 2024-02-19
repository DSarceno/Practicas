SELECT cou.name AS PAIS,
	   lea.name AS LIGA,
	   m.season AS TEMPORADA,
	   m.stage AS JORNADA,
	   m.date AS FECHA,
	   tea.team_long_name AS EQUIPO_LOCAL,
	   tea.team_short_name AS EQUIPO_LOCAL_MOTE,
	   t.team_long_name AS EQUIPO_VISITANTE,
	   t.team_short_name AS EQUIPO_VISIANTE_MOTE,
	   m.home_team_goal AS GOLES_EQUIPO_LOCAL,
	   m.away_team_goal AS GOLES_EQUIPO_VISIANTE,
	   CASE
           WHEN m.home_team_goal > m.away_team_goal THEN 'W'  -- Si el equipo local gana
           WHEN m.home_team_goal < m.away_team_goal THEN 'L'  -- Si el equipo local pierde
           ELSE 'D'                                           -- Si hay empate
       END AS RESULTADO_EQUIPO_LOCAL
FROM Match AS m
INNER JOIN Country AS cou ON cou.id = m.country_id
INNER JOIN League AS lea ON lea.id = m.league_id
INNER JOIN Team AS tea ON tea.team_api_id = m.home_team_api_id
INNER JOIN Team AS t ON t.team_api_id = m.away_team_api_id
ORDER BY m.season ASC