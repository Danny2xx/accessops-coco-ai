from manim import *


class WordEmbedding(Scene):
    def construct(self):
        self.part1_word_to_vector()
        self.part2_clustering()

    # ─────────────────────────────────────────────────────────────
    # PART 1 — Word → Vector  (0:22–0:38 approx)
    # ─────────────────────────────────────────────────────────────
    def part1_word_to_vector(self):
        title = Text("The Embedding Layer", font_size=40, color=WHITE)
        subtitle = Text(
            "Every word becomes a vector of 256 numbers",
            font_size=22,
            color=GRAY,
        )
        subtitle.next_to(title, DOWN, buff=0.25)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(subtitle), run_time=0.5)
        self.wait(0.5)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.4)

        # ── Words on the left ──────────────────────────────────
        word_list   = ['"dog"', '"cat"', '"running"', '"walking"', '"a"']
        word_colors = [BLUE_C, BLUE_C, GREEN_C, GREEN_C, GRAY]

        word_mobs = VGroup(*[
            Text(w, font_size=30, color=c)
            for w, c in zip(word_list, word_colors)
        ]).arrange(DOWN, buff=0.42).shift(LEFT * 3.8)

        self.play(
            LaggedStart(*[FadeIn(w, shift=RIGHT * 0.3) for w in word_mobs],
                        lag_ratio=0.15),
            run_time=1.2,
        )
        self.wait(0.2)

        # ── Arrows + vector strings ────────────────────────────
        vec_strings = [
            "[ 0.21, −0.83,  0.44, … ]",
            "[ 0.19, −0.79,  0.51, … ]",
            "[ −0.62,  0.34, −0.11, … ]",
            "[ −0.58,  0.29, −0.08, … ]",
            "[ 0.03, −0.02,  0.01, … ]",
        ]

        arrow_group  = VGroup()
        vector_group = VGroup()

        for word_mob, vec_str in zip(word_mobs, vec_strings):
            start = word_mob.get_right() + RIGHT * 0.18
            end   = start + RIGHT * 2.3

            arrow = Arrow(
                start, end,
                buff=0,
                stroke_width=2.5,
                color=GRAY_B,
                max_tip_length_to_length_ratio=0.12,
            )

            vec_text = Text(vec_str, font_size=17, color=YELLOW_A)
            vec_text.next_to(arrow.get_end(), RIGHT, buff=0.12)
            vec_text.align_to(word_mob, UP).shift(DOWN * 0.04)

            arrow_group.add(arrow)
            vector_group.add(vec_text)

        self.play(
            LaggedStart(*[
                AnimationGroup(GrowArrow(a), FadeIn(v, shift=RIGHT * 0.2))
                for a, v in zip(arrow_group, vector_group)
            ], lag_ratio=0.18),
            run_time=2.0,
        )

        # ── "×256 dimensions" badge ────────────────────────────
        dim_note = Text("× 256 dimensions", font_size=21, color=YELLOW)
        dim_note.next_to(vector_group, DOWN, buff=0.35)
        badge = SurroundingRectangle(dim_note, color=YELLOW, buff=0.14,
                                     corner_radius=0.1)

        self.play(Write(dim_note), Create(badge), run_time=0.7)
        self.wait(0.4)

        bottom = Text(
            "30,000 tokens.  256 numbers each.  Learned — not programmed.",
            font_size=20,
            color=WHITE,
        )
        bottom.to_edge(DOWN, buff=0.45)
        self.play(Write(bottom), run_time=1.0)
        self.wait(1.2)

        self.play(
            FadeOut(VGroup(word_mobs, arrow_group, vector_group,
                           dim_note, badge, bottom)),
            run_time=0.6,
        )

    # ─────────────────────────────────────────────────────────────
    # PART 2 — Clustering  (0:38–0:52 approx)
    # ─────────────────────────────────────────────────────────────
    def part2_clustering(self):
        title2 = Text("Similar words → similar vectors", font_size=34, color=WHITE)
        title2.to_edge(UP, buff=0.45)
        self.play(Write(title2), run_time=0.7)

        axes = Axes(
            x_range=[-4.5, 4.5, 1],
            y_range=[-3.2, 3.2, 1],
            x_length=9,
            y_length=5.6,
            axis_config={"color": DARK_GRAY, "stroke_width": 1.5},
            tips=False,
        ).shift(DOWN * 0.35)

        self.play(Create(axes), run_time=0.5)

        # ── Cluster data ───────────────────────────────────────
        animal_data = [
            ("dog",    -2.2,  1.4),
            ("cat",    -2.7,  0.9),
            ("puppy",  -1.9,  1.9),
            ("kitten", -2.5,  2.0),
        ]
        movement_data = [
            ("running",  2.3, -1.2),
            ("walking",  2.8, -0.7),
            ("jumping",  1.9, -1.7),
        ]

        def build_cluster(data, color):
            dots   = VGroup()
            labels = VGroup()
            for word, x, y in data:
                pos   = axes.c2p(x, y)
                dot   = Dot(pos, color=color, radius=0.13)
                label = Text(word, font_size=22, color=color)
                label.next_to(dot, UP, buff=0.09)
                dots.add(dot)
                labels.add(label)
            return dots, labels

        a_dots, a_labels = build_cluster(animal_data,   BLUE_C)
        m_dots, m_labels = build_cluster(movement_data, GREEN_C)

        self.play(
            LaggedStart(*[
                AnimationGroup(FadeIn(d, scale=0.4), Write(l))
                for d, l in zip(a_dots, a_labels)
            ], lag_ratio=0.25),
            run_time=1.5,
        )
        self.wait(0.2)
        self.play(
            LaggedStart(*[
                AnimationGroup(FadeIn(d, scale=0.4), Write(l))
                for d, l in zip(m_dots, m_labels)
            ], lag_ratio=0.25),
            run_time=1.2,
        )

        # ── Circles around each cluster ────────────────────────
        a_circle = Ellipse(width=2.2, height=2.0, color=BLUE_C,  stroke_width=2.5)
        a_circle.move_to(axes.c2p(-2.33, 1.55))

        m_circle = Ellipse(width=2.4, height=1.6, color=GREEN_C, stroke_width=2.5)
        m_circle.move_to(axes.c2p(2.35, -1.2))

        self.play(Create(a_circle), Create(m_circle), run_time=0.9)

        # ── Final note ─────────────────────────────────────────
        final = Text(
            "Learned during training.  Not hand-crafted.",
            font_size=22,
            color=WHITE,
        )
        final.to_edge(DOWN, buff=0.4)
        self.play(Write(final), run_time=0.8)
        self.wait(1.8)
