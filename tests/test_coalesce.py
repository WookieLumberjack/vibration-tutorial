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
