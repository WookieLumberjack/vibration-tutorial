"""`gui/coalesce.py`: a page's redraws asked for while a trial is set up run once, at the end (#409 P-7). A tab
shown mid-trial: `test_lazy_tabs.py`."""

from vib_tutorial.gui import coalesce


class Page:
    def __init__(self) -> None:
        self.drawn: list[object] = []

    @coalesce.coalesced
    def redraw(self, what: object = None) -> None:
        self.drawn.append(what)


def test_a_redraw_runs_at_once_outside_a_batch_and_once_at_the_end_of_one():
    p, q = Page(), Page()
    p.redraw(0)
    assert p.drawn == [0]
    with coalesce.batch():
        p.redraw(1)
        q.redraw("q")
        with coalesce.batch():  # nested: only the outermost's end draws
            p.redraw(2)
        assert p.drawn == [0] and q.drawn == [] and coalesce.active()
    assert p.drawn == [0, 2] and q.drawn == ["q"]  # once each, with the last call's arguments
    assert not coalesce.active()


def test_deferred_calls_run_in_the_order_last_asked_and_may_ask_for_more():
    order: list[str] = []
    with coalesce.batch():
        coalesce.defer(order, "a", lambda: order.append("a"))
        coalesce.defer(order, "b", lambda: (order.append("b"), coalesce.defer(order, "c", lambda: order.append("c"))))
        coalesce.defer(order, "a", lambda: order.append("a"))  # asked again: after b
    assert order == ["b", "a", "c"]


def test_a_view_asked_by_its_page_and_by_what_the_page_asks_for_draws_once_after_them():
    log: list[str] = []

    class View:
        @coalesce.coalesced(leaf=True)
        def draw(self) -> None:
            log.append("view")

    class Playback:
        def __init__(self, view: View) -> None:
            self.view = view

        @coalesce.coalesced
        def place(self) -> None:
            log.append("place")
            self.view.draw()

    view = View()
    playback = Playback(view)
    with coalesce.batch():
        view.draw()  # the page's redraw sets the view up
        playback.place()  # and its playback places the body
    assert log == ["place", "view"]
    view.draw()
    assert log == ["place", "view", "view"]  # outside a batch: at once


def test_redraws_that_keep_asking_for_each_other_fail_rather_than_hang():
    import pytest

    class Loop:
        @coalesce.coalesced
        def a(self) -> None:
            self.b()

        @coalesce.coalesced
        def b(self) -> None:
            self.a()

    with pytest.raises(RuntimeError, match="asking for each other"):
        with coalesce.batch():
            Loop().a()
    assert not coalesce.active()


def test_a_slot_given_more_than_it_takes_drops_the_rest_as_pyside_does():
    class Tab:
        def __init__(self) -> None:
            self.drawn = 0

        @coalesce.coalesced
        def draw(self) -> None:
            self.drawn += 1

    t = Tab()
    t.draw(3)  # a tab's index, from currentChanged
    with coalesce.batch():
        t.draw(4)
    assert t.drawn == 2
