# Moko growth notes (read by the nightly run before writing episodes)

Updated every Monday by the weekly review. Newest notes on top.

## 2026-11-09 · weekly review #2 (week 2–8 Nov)
- BLOCKER: Metricool hit its account posting limit. Every post from 6 Nov 19:00 onward (ep10–ep14 and the 8 Nov compilation) FAILED on YouTube and Facebook with "You have reached your Metricool account limit". Only ep01–ep09 actually went live. Kinshuk has been alerted. Until he confirms the limit is fixed, keep making and logging episodes as normal, but check getScheduledPosts each night: if earlier posts show status ERROR, mark them "FAILED – Metricool limit" in EPISODE_LOG.md instead of "scheduled", and treat failed episodes as never shown (their topics can be reused later).
- Numbers so far (data lags): YouTube ~9 views total, 33 subscribers (0 gained). Facebook reels 741 views on 9 reels, followers 5 → 6. Watch-through is weak: Facebook average view is only 2–7 s on 25–40 s reels.
- Timing test NOT possible yet: no 10:00/16:00 post has gone live. Keep 10:00/16:00 and compare next week. Baseline from old slots on Facebook: 07:00 reels averaged ~129 views vs ~24 at 19:00, so morning looks much stronger than evening.
- Best Facebook reels by views: "गिनती! एक दो तीन" (224), "मोको को आया गुस्सा! शेयर करना" (199), "मोको खो गया! बड़ों के साथ" (127). Worst: bedtime "नींद नहीं आई" (15), hand-washing (16), brushing / please-thank-you (28 each). Best watch time: hand-washing and road safety (7 s avg). Bedtime and colour episodes did worst on both.
- Rule: open with a strong problem and big emotion in the first 2 s (angry, scared, lost, counting challenge). Calm or sleepy openings (bedtime, calm breathing) lose viewers; if you make a calm episode, still start with a sudden problem.
- Rule: keep title style "मोको + strong feeling/problem! + emoji + lesson" (गुस्सा, डर गया, खो गया). Avoid soft titles with no "!" like "मोको को नींद नहीं आई".
- Rule: next week give more space to counting/numbers (best reach), feelings conflicts (sharing, anger, fear) and safety with Grandpa Wind. Move bedtime to at most 1 episode a week until it improves.
- Rule: fight drop-off. Get to the lesson by about 8 s, keep episodes nearer 25–30 s than 40 s, and end with a quick repeat line kids can say along ("एक, दो, तीन!") so the reel loops.
- vidIQ credits: 30 left (reset 10 Nov 18:20 UTC). Use no paid vidIQ calls; reuse the starting keywords.

## 2026-11-05 · manual note (from Kinshuk's Studio check)
- Day 4: ~13 views in total, "stayed to watch" 47%. Problem is distribution, not the videos. Since ep13 the engine adds Hindi subtitles and a big hook title in the first 2.8 s (meta.title text before the first "!" or "?", or meta.hook to override).
- Write title text so its first sentence works as the on-screen hook (short, emotional, ends with "!" or "?").

## 2026-11-02 · weekly review #1 (data too early)
- Week 26 Oct–1 Nov: no videos were live yet (first episodes go out 2 Nov), so Metricool shows 0 views. YouTube starts at 33 subscribers, Facebook at 5 followers. No lessons from data yet.
- Keep ALL starting rules below unchanged for this week.
- Post both 07:00 and 19:00 IST every day so next Monday can compare the two time slots.
- Rotate pillars evenly (feelings, healthy habits, safety, learning numbers/colours, bedtime) so next week shows which pillar wins.
- vidIQ credits are low (65 left, reset 10 Nov): use vidIQ only where really needed; reuse the keywords listed below instead of new keyword research.

## 2026-11-02 · starting rules
- Channel is new: focus on consistency (2 Shorts a day, 07:00 and 19:00 IST) and one compilation every Sunday 11:00 IST.
- Strongest search keywords (vidIQ, India): "hindi cartoon story", "kids story in hindi", "hindi moral stories", "cartoon stories in hindi".
- Hook in the first 2 seconds: Moko's weather should change at once (sudden rain, snow, grey cloud) so kids keep watching.
- Prefer topics parents search for: hand washing, brushing teeth, sharing, saying sorry, road safety, counting 1–10, colours, bedtime.
- Every episode MUST set "location" in spec.json (sky, park, town, farm, road, night, school, garden). Never use the same location two episodes in a row; match the topic (road = road safety, night = bedtime, garden/farm = food and nature, school = letters/numbers). Use one "go:<location>" change in about half the episodes so the picture changes mid-story.
- Visual variety matters for monetization: YouTube can refuse to monetize channels whose videos all look like one template. Vary location, props, characters and moods across the week.
- Keep episodes 25–40 seconds; Shorts that are watched to the end grow fastest.
